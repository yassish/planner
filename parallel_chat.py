import json
import time
import asyncio
import requests
import concurrent
import numpy as np
import pandas as pd
import concurrent.futures
from tqdm.auto import tqdm
from datetime import datetime
import argparse
import random
from transformers import set_seed
from datasets import load_dataset

import torch
from torch.utils.data import DataLoader, Dataset
import utils
import os


class APIModel():
    def __init__(self,
                 model_name,
                 api_key_path='./keys.json',
                 timeout=200,
                ):
        with open(api_key_path, 'r') as file:
            api_keys = json.load(file)
        self.api_key = api_keys["unified_api_key"]
        self.endpoint = 'http://54.186.24.124:5003/llm_unified_api'
        self.model = model_name
        self.timeout = timeout

    def generate(self, messages, temperature=0.7, max_tokens=1024, generation = 1):
        responses = []
        for i in range(generation):
            for sleep_time in [1, 2, 4]:
                try:
                    response = requests.post(
                    self.endpoint, 
                    headers={'api-key': self.api_key, 'Content-Type': 'application/json'},
                    json={
                        'messages': messages,
                        'model_option': self.model,
                        'temperature': temperature,
                        'max_tokens': max_tokens,
                    }, 
                    timeout=self.timeout
                )
                    if response.ok:
                        response_json = response.json()
                        if response_json.get('llm_api_call_success') == 'True':
                            responses.append(response_json.get('response'))
                            break
                except requests.RequestException as e:
                    print(f"Request failed: {e}")
                except ValueError as e:
                    print(f"Failed to parse JSON response: {e}")

                time.sleep(sleep_time)

            #raise Exception("Request failed after retries")
            if len(responses) == 0:
                raise Exception(response.json())
        return self.model, responses

        
        
def step(prompt, config, prev_response=None):
    """Get the next generation of all of the reference_models in parallel"""
    
    # define api models
    api_models = []
    for model in config['reference_models']:
        api_models.append(APIModel(model))
    
    # run models in parallel
    results = []
    num_parallels = len(api_models)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallels) as executor:
        future_to_url = {}
        for layer_id in range(len(layers)):
            def fetch_response():
                # Define fetch_response for each api_model separately
                if not prev_response:
                    messages = [
                            {"role": "user",
                             "content": prompt}
                        ]
                    generation = 2
                    temperature = config['temperature_first_layer']
                else:
                    messages = [
                            {"role": "system",
                             "content": utils.getFinalSystemPrompt(config['layer_prompts'][layer_id], prev_response)},
                             {"role": "user",
                         "content": prompt}
                        ]
                    generation = 1
                    temperature = config['temperature_next_layer']
                model_name, response = api_model.generate(
                        messages,
                        temperature=temperature,
                        max_tokens=config['max_tokens'], 
                        generation = generation
                    )
                return response

            # Use the current messages for the current api_model
            future = executor.submit(fetch_response)
            future_to_url[future] = api_model.model  # It might be more useful to map futures to models, not messages
        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(future_to_url):
            data = future.result()
            results.extend(data)
    
    return results




def prepare_prompt(raw_prompt):
    prompt = f'''\
You are tasked with completing a coding problem. You will be given a function name, \
its inputs, and its docstring. Please complete it as best as you can. Please be \
concise only complete the code without any extra explanation. Here is a problem

```python
{raw_prompt}
'''
    return prompt
        
def one_problem(prompt, config):
    
    results = step(prompt, config)
    
    for _ in range(1, config['layers'] - 1):
        results = step(prompt, config, prev_response=results)
      
    model = APIModel(config['model'])
    dut_generation_prompt = f'''Integrate teh submodules into one working module in SystemVerilog, remebering the description:
    =====
    {prompt}
    =========
    
    Your generated code should be like
    
    module dut (...);
       ...
    endmodule
    
    The output must be a YAML object equivalent to type $ProblemSolutions, according to the following Pydantic definitions:
    ======
    class Solution(BaseModel):
        verilog: str = Field(description="generated code")


    class $ProblemSolutions(BaseModel):
        possible_solutions: List[Solution] = Field(max_items=1, description="A list of possible solutions to the problem. Make sure each solution fully addresses the problem rules and goals.")
    ======


    Example YAML output:
    ```yaml
    solution:
    - verilog: |
        ...
     ```

    Answer:
    ```yaml\

    '''
    
     
    final_messages=[
        {"role": "user",
         "content": utils.getFinalSystemPrompt(dut_generation_prompt, results)},
        # {"role": "user",
        #  "content": prompt}
    ]
    model_name, final_response = model.generate(
        final_messages,
        temperature=config['temperature'],  #config['temperature'],
        max_tokens=config['max_tokens']
    )
    
    
    try:
        solution = utils.load_yaml(final_response[0],
                                keys_fix_yaml=["verilog:"])['solution']

    except Exception as e: 
        raise f"No solutionfor {description}: {e}"
        
    return solution
def generate_responses(config, df):
    
    model = APIModel(config['model_name'])
    
    print('\n======= Creating responses =======\n')
    to_run = []
    for k in tqdm(range(1, config['num_passes']+1)):
        if f'response_{k}' in df.columns and not sum(df[f'response_{k}'].isna()):
            print(f'pass {k} already ran!')
        else:
            to_run.append(k)


    for k in to_run:
        # if f'response_{k}' in df.columns:
        #     print(f'pass {k} already ran!')
        # else:   
        print(f"---------------pass {k} --------------")
        results = run_problems(df, config)
    
        for item in results:
            df.loc[item.name, f'response_{k}'] = item['completion']
        rows_list = []
        for index, row in df.iterrows():
            # Convert row to dictionary
            row_dict = row.to_dict()
            # Append dictionary to list
            rows_list.append(row_dict)



        # Save the list of dictionaries to a JSON file
        with open(config['solution_path'], 'w') as json_file:
            json.dump(rows_list, json_file, indent=4)

        print(f"Rows saved to {config['solution_path']}")


    print(f"\n======= Generation of {config['num_passes']} passes for {len(df)} questions is done. =======\n")
def prepare_messages(question, system_prompt):
    # TODO: study whether it is better to have the code marker at the end or not
    #query = prompt + f'''### QUESTION\n{question.lstrip().strip()}\n\n''' #question #+ '\nBegin \n```verilog\n'
    
    format_prompt = f'''
       You are an expert SystemVerilog programmer. When generating SystemVerilog code, please adhere to the following guidelines to ensure the code fully utilizes SystemVerilog features and avoids common mistakes associated with confusing it with Verilog.
- Only output the code snippet and do NOT output anything else.
- If module name is not specified, call it "dut".
- If the name is not specified for the clock port, it should be called “clock”.
- If polarity is not specified, signals should be active high.
- If multiple clock domains exist, the name of the domain should be added as a suffix to the clock and reset (i.e. clockRead, clockWrite, resetRead, resetWrite).
- Provide the minimum functionality that meets the request, do not add additional features, especially no additional ports.
- If a sequential circuit is not required, do not include clock or reset inputs. A sequential circuit requires a reset when it has internal state that feeds back into itself.

    Here is the problem description:
    ======
    {question}
     ======

    Answer the questions regards to the description'''
    
     
    
    
    
    query = format_prompt + prompt
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
    else:
        messages = [
            {"role": "user", "content": query},
        ]
    return messages
                        
def run_problems(raw_dataset, config):
    
    final_results = []
    
    output_path = os.path.join(config['result_output_dir'], f"{config['result_name']}.csv")
    with concurrent.futures.ThreadPoolExecutor(max_workers=config["num_parallel_calls"]) as outer_executor:
        
        def get_response(item):
            prompt = prepare_messages(item['description'], config['system_prompt'])
            response = one_problem(prompt, config)
            num_passes = config['num_passes']
            result = {}
            result['index'] = item['task_id']
            result[f'response_{num_passes}'] = response
            result['pre_prompt'] = item['pre_prompt']
            result['raw_prompt'] = item['raw_prompt']
            result['target'] = item['canonical_solution']
            result['test'] = item['test']
            result['func_name'] = item['func_name']
            
            #result = {'index': item['task_id'], 'response': response}
            return result
        
        future_to_batch = {}
        for item in raw_dataset:
            future = outer_executor.submit(get_response, item)
            future_to_batch[future] = item
        
        for future in tqdm(
            concurrent.futures.as_completed(future_to_batch),
            total=len(raw_dataset),
            desc="Creating Responses"
        ):
            item_results = future.result()
            final_results.append(item_results)
            
    df = pd.DataFrame(final_results)
    df.to_csv(os.path.join(config['result_output_dir'], f"{config['result_name']}.csv"))
    return final_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SystemVerilog planner")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration JSON file")
    args = parser.parse_args()
#     # # Step 1: Run the model to generate results and config
#     print("Running model to generate results and config...")
    config = utils.load_json(args.config)
    
    if 'seed' in config.keys():
        set_seed(config['seed'])
#     #print(config)
    dataset = load_dataset(config["evaluation"])['test']
    final_results = run_problems(dataset.select(np.arange(config['subsample_size'])), config)
    #result = moa_one_problem(prompt, config)
    #print(result)

    for item in final_results:
        print('\n=========\n')
        print(item['response'])
    

