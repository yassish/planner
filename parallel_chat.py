"""
SystemVerilog Code Generator using LLM API

This script generates SystemVerilog code using a Language Model API. It processes input descriptions
and generates corresponding SystemVerilog implementations through a multi-layer generation approach.

Major components:
- APIModel: Handles API communication and response generation
- Problem processing: Converts descriptions into SystemVerilog code
- Concurrent execution: Processes multiple problems in parallel

Requirements:
- Python 3.7+
- Required packages: torch, transformers, pandas, numpy, requests, tqdm
"""

import json
import time
import asyncio
import requests
import concurrent.futures
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
import argparse
import random
import os
from typing import List, Dict, Any, Optional
from transformers.trainer_utils import set_seed
from datasets import load_dataset
# import torch
# from torch.utils.data import DataLoader, Dataset
import utils
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIModel:
    """
    Handles communication with the LLM API for code generation.
    
    Attributes:
        config (dict): Configuration parameters for the model
        api_key (str): API authentication key
        endpoint (str): API endpoint URL
        model (str): Name of the model to use
        timeout (int): Request timeout in seconds
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        api_key_path: str = './keys.json',
        timeout: int = 60
    ):
        try:
            with open(api_key_path, 'r') as file:
                api_keys = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"API key file not found at {api_key_path}")
            
        self.config = config
        self.api_key = api_keys.get("unified_api_key")
        if not self.api_key:
            raise ValueError("API key not found in keys file")
            
        self.endpoint = 'http://54.186.24.124:5003/llm_unified_api'
        self.model = config['model_name']
        self.timeout = timeout

    def generate(self, messages: List[Dict[str, str]], generation: int = 1) -> tuple:
        """
        Generate responses using the LLM API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            generation: Number of responses to generate
            
        Returns:
            tuple: (model_name, list of responses)
        """
        # responses = []
        retry_delays = [1, 2, 4]  # Exponential backoff
        response_text = ''
        # for _ in range(generation):
        # print(messages)
        for sleep_time in retry_delays:
            try:
                response = requests.post(
                    self.endpoint,
                    headers={
                        'api-key': self.api_key,
                        'Content-Type': 'application/json'
                    },
                    json={
                        'messages': messages,
                        'model_option': self.model,
                        'temperature': self.config['temperature'],
                        'max_tokens': self.config['max_tokens'],
                    },
                    timeout=self.timeout
                )
                
                if response.ok:
                    response_json = response.json()
                    # print(response_json)
                    if response_json.get('llm_api_call_success') == 'True':
                        
                        response_text = response_json.get('response')
                        # messages = response_json.get('messages')
                        break

            except requests.RequestException as e:
                logger.error(f"Request failed: {e}")
            except ValueError as e:
                logger.error(f"Failed to parse JSON response: {e}")

            time.sleep(sleep_time)

        if not response_text:
            try:
                error_details = response.json()
            except:
                error_details = "Unknown error"
            raise Exception(f"API request failed after all retries. Details: {error_details}")
        # print(json.loads(messages))      
        return self.model, response_text, messages

def step(prompt: str, config: Dict[str, Any], layer_id: Optional[int] = None, prev_response: Optional[str] = None) -> List:
    """
    Execute one step of the generation process.
    
    Args:
        prompt: Input prompt for generation
        config: Configuration dictionary
        prev_response: Previous step's response if any
        
    Returns:
        List of generated responses
    """
    model = APIModel(config)
    messages = []
    if prev_response:
       
        messages.append({
            "role": "user",
            "content": prev_response + config['layer_prompts'][layer_id] + '\n [NEW STEP]\n '
                                                 
        })
    else:
        messages.append({
            "role": "user",
            "content": prompt + config['layer_prompts'][layer_id] + '\n [NEW STEP]\n '
        })
    _, response, previous_messages = model.generate(
        messages
    )
    return  (previous_messages[-1]['content'] + 
            '\n' +
            response)
def fix_code_from_tests_failure(description, solution, error, config ):

    fix_code_prompt=f"""\

        problem description:
        ================
        {description}
        ================

        A genertaed code:
        =============
        {solution}
        =============


        However, when running the test, the code solution failed to run with this error:
        =============
        {error}
        =============

        Using the information above, your goal is to generate a fixed code, that will correctly solve the error. 
        return the complete module dut and do not duplicate any definition. The final code should include all modules originally provided. 
        Do not include any refence to *.v files in the testbench. tb_control module is responsible for managing the overall control flow of the testbench. 
    
        The output must be a YAML object equivalent to type $FixedCode, according to the following Pydantic definitions:
        =====
        class FixedCode(BaseModel):
            fixed_code: str = Field(description="A fixed code solution. Don't explain your answer. Just provide a fixed code, and nothing else")
        =====

        Example YAML output:
        ```yaml
        fixed_code: |-
        ...
        ```
        Each YAML output MUST be after a newline, indented, with block scalar indicator ('|-').

        Answer:
        ```yaml"""
    messages = (
                [  
                    {"role": "user", "content": fix_code_prompt},
                ]
            )
    model = APIModel(config)
    
    _, response_fixed_code, messages = model.generate(
        messages
    )

    try:
        response_fixed_code = utils.load_yaml(response_fixed_code,keys_fix_yaml=["fixed_code:"])['fixed_code']
    
        response_fixed_code = response_fixed_code.rstrip("'` \n") # remove trailing spaces and newlines from yaml response

        if response_fixed_code.startswith("```systemverilog"):
            response_fixed_code = response_fixed_code[13:]
        solution = response_fixed_code
    except Exception as e:
        print(f"Failed to parse solution: {e}")
    return solution


def one_problem(prompt: str, config: Dict[str, Any], guides: Optional[str]= None) -> Dict:
    """
    Process a single problem through all generation layers.
    
    Args:
        prompt: Problem description
        config: Configuration dictionary
        
    Returns:
        Generated solution dictionary
    """
    results = step(prompt, config, 0)
    # print(results)
    for lid in range(1,config.get('layers', 1)-1):
        results  = step(prompt, config, lid, prev_response=results)
        # print(results)
    # print(results)
    model = APIModel(config)
    dut_generation_prompt = create_dut_prompt(prompt, guides)
    
    final_messages = [{
        "role": "user",
        "content": utils.getFinalPrompt(dut_generation_prompt, [results])
    }]
    
    _, final_response, messages = model.generate(
        final_messages
    )
    # print(final_response)
    try:
        solution = utils.load_yaml(
            final_response,
            keys_fix_yaml=["verilog:"]
        )['solution'][0]['verilog']
        # print(solution)
    except Exception as e:
        logger.error(f"Failed to parse solution: {e}")
        raise ValueError(f"No solution for description: {e}")
        
    return solution, messages[-1]['content']

def create_dut_prompt(description: str, guides: Optional[str]) -> str:
    """Create the DUT generation prompt with proper formatting."""
    with open('guidelines.txt', 'r') as file:
        guides =  file.read()
    guides = f"""{guides}""" 
    
    return  f'''
    Integrate the submodules into one working module in SystemVerilog. Include all submodules in your final answer as one workingf module. Leave no comment referencing previous answers. Remembering the description:
    =========
    {description}
    =========
    
    Your generated code should work standlone without any import of libraries, and should include reference to any instanstiation. Make sure to check parametres and pins for all modues and their instantiation. Here are some general guidelines:
   ==========
   {guides}
   ==========
   The code should be inside a module called `dut`:
    
    module dut (...);
       ...
    endmodule
    
    The output must be a YAML object equivalent to type $ProblemSolutions, according to the following Pydantic definitions in $ProblemSolutions:
    ======
    class Solution(BaseModel):
        verilog: str = Field(description="generated code")


    class $ProblemSolutions(BaseModel):
        solution: List[Solution] = Field(max_items=1, description="A list of possible solution to the problem. Make sure each solution fully addresses the problem rules and goals.")
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

def prepare_tests_for_api_from_df(df_source):
    """
    A function that prepares rows of a dataframe for verilator test API
    """
    # fill in empty fields to avoid error on the API side
    # df_source = df_source.fillna('This field was empty!')
    tests = []
    for i,row in df_source.iterrows():
        item = row.to_dict()
        tests.append( json.dumps( item ) )

    return tests
def prepre_df_to_test(df_dict, sol_column = 'dut', test_columns = ['tb']):
        # # Assuming df_dict is defined and data is initialized
        data = []
        random_selected = []
        for k, v in df_dict.items():
            # Get the generated solutions and tests once
            generated_solutions = v[sol_column]
            generated_tests = []
            
            tb = v['tb']
            name = v['name']
            
            data.extend([
            {
                'solution_id': idx_sol,
                'dut': solution,
                'index': int(k),
                'tb': tb,
                'name': name
            }
            for (idx_sol, solution) in product(enumerate(generated_solutions))
            ])
        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(data)
        
def fix_code_if_failed(results, df_source, config):
    
    for i,row in results.iterrows():
                        #logger.debug(f"test pass flag {results['pass'].values[0]}")
        if row['pass']:
            # scores.append(True)
            print(f'problem {row['name']}  has passed the test')
            print(f'-'*30)
        else: 
            # scores.append(False)
            #fix_code(results['stderr'].values[0])
            error = row['stderr']
            print(f'problem {row['name']}  has the error:  \n {error}')
            calls = 0
            #logger.info(results['stdout'].values[0])
            while ("%Error" in error or "%Warning" in error) and calls < config['max_calls']:
                # flags[i] = False


                #while "%Error: Cannot continue" in error and calls <= self.config['max_calls']:
                print(f'-'*30)
                print(f'regenerating the code as it had error')
                query = df_source[df_source['index'] == row['index']]['query'].values[0]
                solution = fix_code_from_tests_failure(query, row['dut'], error,config)
                df = results.loc[[i],['index','tb','name']].copy()
                df['dut'] = solution
                # print(df.head())
                result = run_testbench_batch(df,cols = ['dut',  'tb'],
                                                        solution_column = 'dut', test_column = 'tb', copy_all_fields=True)
                error = result['stderr'].values[0]
                # print(f'new test error {error}')
                calls += 1
                if result['pass'].values[0] or error == '':
                    # flags[i] = True
                    # scores.append(True)
                    results.loc[i,'dut'] = solution
                    print(f'passed the test after {calls} calls to generate')
                    print(f'-'*30)

            print(f'failed the test after {calls} calls to re-generate')
            print(f'-'*30)
    return results
            
            
                    
def run_test_in_batches(df_test, df_source, config, sol_column = 'dut', test_columns = ['tb']):

        # df_source = pd.DataFrame.from_dict(df_dict, orient='index')
        batch_size = 16
        for test_col in test_columns:
            
            # df = prepre_df_to_test(df_dict, sol_column = sol_column ,test_columns = [test_col] )

            n_batches = int(np.ceil(len(df_test)/batch_size))
            print(f"number of batches {n_batches}")
            # logger.info('Choose best solution')
            results = []
            if n_batches > 0:
                for i in tqdm(range(n_batches)):
                    
                    df_batch = df_test[i*batch_size:min((i+1)*batch_size,len(df_test))].copy()
                    result = run_testbench_batch(df_batch,cols = ['dut',  test_col],
                                                        solution_column = 'dut', test_column = test_col, copy_all_fields=True)
                    result = fix_code_if_failed(result, df_source, config)
                    
                    results.append(result)

                results  = pd.concat(results)
        return results

def run_testbench_batch(df, cols = ['dut',  'tb'], solution_column = 'dut', test_column = 'tb', test_type = None, copy_all_fields = False):
        #df = df.sort_values(by=['index', 'solution_id'])
        cols = ['index','name'] + cols
        df_source= df[cols].copy() 
        df_source.rename(columns = {solution_column:'dut',  test_column: 'tb'}, inplace = True)
        tests = prepare_tests_for_api_from_df(df_source)
        #logger.info(tests)
        compile_only = False
        if test_type == 'ai':
            compile_only = True
        with open('./keys.json', 'r') as file:
            keys = json.load(file)
        
        api = keys["unified_api_key"] #"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" #keys["unified_api_key"]
        verilator_endpoint = 'http://34.223.52.212:5005/test_runner_api'
        # TODO: is this the format we want or we want to explicitely determine the tools in config?
        
        response = requests.post(
            verilator_endpoint, 
            headers={'api-key': api, 'Content-Type': 'application/json'},
            json={
                'tests': tests,
                'sim_tool': "verilator", 
                'compile_only' : compile_only,
                'copy_all_fields': copy_all_fields
            },
            timeout=2000
        )

            
        if not response.ok:
            print('The api failed!')
            print(response.content)
            return
        #logger.info(response.content)
        results = json.loads(response.content)['results']
        #logger.info(results)
        assert len(results) == len(tests)
        results_df = str_results_to_df(results)

        return results_df
    
    
    
def run_problems(raw_dataset: pd.DataFrame, config: Dict[str, Any]) -> List[Dict]:
    """
    Process multiple problems concurrently.
    
    Args:
        raw_dataset: DataFrame containing problems to process
        config: Configuration dictionary
        
    Returns:
        List of results for each problem
    """
    
    # print(len(raw_dataset))
    date = datetime.now().date()
    for i in range(config["num_passes"]):
        # final_results = []
        config['solution_path'] = os.path.join(
            './outputs',
            f"{config['dataset']}_{config['model_name']}_{date}_{i}.json"
        )
        config_path = f"./configs/{config['dataset']}_{config['model_name']}_{date}_{i}.json"
        with open(config_path, 'w') as json_file:
            json.dump(config, json_file, indent=4)
        # to_run = []    
        try:
            # print(config['solution_path'])
            data = utils.load_json(config['solution_path'])
            # print(data[0]['index'])
            to_run = [int(k['index']) for k in data]
            print(f' indices alreday ran {to_run}')
            raw_dataset = raw_dataset[~raw_dataset['index'].isin(to_run)]
        except:
            data = []
            print(f'no file saved with name {config['solution_path']}. Generating from the top!')
        
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=config["num_parallel_calls"]
        ) as executor:
            def get_response(item: Dict) -> Dict:
                # print(item['index'])
                prompt, guides = prepare_messages(item['query'], config.get('system_prompt',None))
                response, history  = one_problem(prompt, config, guides)
                return {
                    'index': item['index'],
                    'dut': response,
                    'conversation': history,
                    'tb': item['tb'],
                    'name': item['name'],
                    # 'guides': guides
                }
        
            futures = {
                executor.submit(get_response, item): item 
                for item in raw_dataset.to_dict('records')
            }
        
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(raw_dataset),
                desc="Creating Responses"
            ):
                
                result = future.result()
                 #Append new data
                data.append(result)
                with open(config['solution_path'], 'w') as json_file:
                    json.dump(data, json_file, indent=4)
    return pd.DataFrame(data), config

def prepare_messages(question: str, system_prompt: Optional[str]) -> List[Dict[str, str]]:
    """
    Prepare messages for the API request.
    
    Args:
        question: Problem description
        system_prompt: Optional system prompt
        
    Returns:
        List of formatted messages
    """
    index = question.find('--------------------------------------------------------------\n')
    format_prompt = f'''Problem description:
    ======
    {question[:index]}
    ======
    
    Answer the questions regarding the description
    '''
    
    # messages = []
    # if system_prompt:
    #     messages.append({"role": "system", "content": system_prompt})
    # messages.append({"role": "user", "content": format_prompt})
    
    return format_prompt, question[index:]


def str_results_to_df(results):
    """
    A function that converts the results of a verilator test API into a df for
    better viewing and analysis
    TODO: This is hacky and needs to be rewritten
    """
    df_result = pd.DataFrame(
        index=range(len(results)),
        columns=json.loads(results[0]).keys()
    )
    for i in df_result.index:
        dictionary = json.loads(results[i])
        for key in dictionary.keys():
            df_result.loc[i, key] = dictionary[key]
    return df_result.sort_values(by='index')

        
        
def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="SystemVerilog code generator")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration JSON file"
    )
    args = parser.parse_args()
    
    try:
        config = utils.load_json(args.config)
        
        
        
        if 'seed' in config:
            set_seed(config['seed'])
        
        df = pd.read_pickle(config['dataset_path'])
        # print(df.head())
        if config.get('selected_indices'):
            df = df.loc[df.index[config['selected_indices']]]
        df = df.reset_index(drop=True)

        # config['solution_path'] = './outputs/demo_google_claude_3.5_sonnet_v2_2024-11-27_0.json'
        
        
        if config.get('solution_path',None):
            results = utils.load_json(config['solution_path'])
            run = [info['index'] for info in results]
            df = df[~df['index'].isin(run)]
        else: 
            print(f'No problem ran for this experiment')
        results, config = run_problems(df, config)
        
        try:
            # print(config['solution_path'])
            results = utils.load_json(config['solution_path'])
            to_test = [v.get('index') for v in results if not v.get('pass', None)]
            results = pd.DataFrame(results)
            
            df_test = results[results['index'].isin(to_test)]

            results = run_test_in_batches(df_test, df, config)
            results = pd.merge(results, df_test, on=['index'], how='inner',suffixes=('', '_right'))
            # Drop the right suffix columns if they exist
            columns_to_drop = [col for col in results.columns if col.endswith('_right')]
            results = results.drop(columns=columns_to_drop)
            print(results.head())
            print('saving the final results')
            rows_list = []
            for index, row in results.iterrows():
                    # Convert row to dictionary
                    row_dict = row.to_dict()
                    # Append dictionary to list
                    rows_list.append(row_dict)
            # Save the list of dictionaries to a JSON file
            with open(config['solution_path'], 'w') as json_file:
                json.dump(rows_list, json_file, indent=4)
        except Exception as e:
            raise f"An error occurred: {e}"

            
    except Exception as e:
        print(f"Script execution failed: {e}")
        raise

if __name__ == '__main__':
    main()