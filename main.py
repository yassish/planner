
"""
System Verilog Code Generator Framework

This module provides a framework for generating code using Language Models and verifying it in batches.
This code generates duts for specific query and batch test versis verilator

Requirements:
- Python 3.7+
- Required packages: requests, pandas, logging, tqdm
"""

import json
import time
import sys
import logging
import requests
import concurrent.futures
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
import os
import utils

# Configure logging
# logging.basicConfig(level=logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(filename)s:%(lineno)d | %(funcName)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Print to console
        #logging.FileHandler(f'logs/run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')  # Save to file
    ]
)
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
        timeout: int = 120
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
                    if response_json.get('llm_api_call_success') == 'True':

                        response_text = response_json.get('response')
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
        logger.debug(f"Failed to parse solution: {e}")
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
    # results = step(prompt, config, 0)
    results = None
    for lid in range(len(config.get('layer_prompts', []))):
        results  = step(prompt, config, lid, prev_response=results)


    dut_generation_prompt = create_dut_prompt(prompt, guides, config['pre_prompt'])

    final_messages = [{
        "role": "user",
        "content": utils.getFinalPrompt(dut_generation_prompt, [results])
    }]
    model = APIModel(config)
    _, final_response, messages = model.generate(
        final_messages
    )

    try:
        solution = utils.load_yaml(
            final_response,
            keys_fix_yaml=["verilog:"]
        )['solution'][0]['verilog']
    except Exception as e:
        logger.error(f"Failed to parse solution: {e}")
        raise ValueError(f"No solution for description: {e}")

    return solution, messages[-1]['content']

def create_dut_prompt(description: str, guides: Optional[str], pre_prompt) -> str:
    """Create the DUT generation prompt with proper formatting."""
    with open('guidelines.txt', 'r') as file:
        guides =  file.read()
    guides = f"""{guides}"""
    prompt = pre_prompt + f'''
     Here is the module description:
    =========
    {description}
    =========

    Your generated code should work standlone without any import of libraries, and should include reference to any instanstiation. Only include the code and no extra information. Make sure to check parametres and pins for all modues and their instantiation. Here are some general guidelines:
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
        solution: List[Solution] = Field(max_items=1, description="A list of possible solution to the problem. Make sure each solution fully addresses the problem rules and goals.Just Provide the code and no extra information.")
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
    return  prompt


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

class BatchVerifier:
    """Handles batch verification of generated code."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.batch_size = config.get('batch_size', 10)
        self.eda_flag = config.get('eda', False)
        self.api_key = self._load_api_key()
        self.verilator_endpoint = config.get('verilator_endpoint', 'http://default-verification-endpoint')

    def _load_api_key(self) -> str:
        with open('./keys.json', 'r') as file:
            return json.load(file)["unified_api_key"]

    def prepare_tests_for_api(self, df: pd.DataFrame) -> List[str]:
        """Prepare dataframe rows for verification API."""
        tests = []
        for _, row in df.iterrows():
            item = row.to_dict()
            tests.append(json.dumps(item))
        return tests

    def verify_batch(self, df_batch: pd.DataFrame, cols: List[str] = ['dut', 'tb', 'tb_name'], eda_flag = True) -> pd.DataFrame:
        """Verify a batch of generated code."""
        df_source = df_batch[['index', 'name'] + cols].copy()
        if eda_flag:
            for i, row in df_source.iterrows():
                df_source.loc[i, 'dut'] = [{"name" : 'dut.sv', "content" : row["dut"]}]

        tests = self.prepare_tests_for_api(df_source)

        try:
            response = requests.post(
                self.verilator_endpoint,
                headers={
                    'api-key': self.api_key,
                    'Content-Type': 'application/json'
                },
                json={
                    'tests': tests,
                    'sim_tool': "verilator",
                    'compile_only': False,
                    'copy_all_fields': True,
                    'eda' : eda_flag
                },
                timeout=2000
            )

            if not response.ok:
                logger.error('Verification API failed!')
                logger.error(response.content)
                return pd.DataFrame()

            results = json.loads(response.content)['results']
            return self._convert_results_to_df(results)

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return pd.DataFrame()

    def _convert_results_to_df(self, results: List[str]) -> pd.DataFrame:
        """Convert API results to DataFrame."""
        df_result = pd.DataFrame(
            index=range(len(results)),
            columns=json.loads(results[0]).keys()
        )
        for i in df_result.index:
            dictionary = json.loads(results[i])
            for key in dictionary.keys():
                df_result.loc[i, key] = dictionary[key]
        return df_result.sort_values(by='index')

    def verify_in_batches(self, df: pd.DataFrame, df_source: pd.DataFrame) -> pd.DataFrame:
        """Process verification in batches."""
        n_batches = int(np.ceil(len(df) / self.batch_size))
        logger.info(f"Processing {n_batches} verification batches")
        results = []
        df = df.set_index('index')
        df.sort_index(inplace=True)
        df_source.sort_index(inplace=True)
        df_source = df_source.set_index('index')
        # print(df_source)
        # print(df)
        df['tb_name'] = df_source['tb_name']
        df.reset_index(inplace=True)
        df_source.reset_index(inplace=True)
        # print(df.head())
        for i in tqdm(range(n_batches)):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, len(df))
            df_batch = df[batch_start:batch_end].copy()

            result = self.verify_batch(df_batch, eda_flag =self.eda_flag)
            if not result.empty:
                result = self.fix_failed_code(result, df_source)
                results.append(result)


        return pd.concat(results) if results else pd.DataFrame()

    def fix_failed_code(self, results: pd.DataFrame, df_source: pd.DataFrame) -> pd.DataFrame:
        """Fix and re-verify failed code."""
        for i, row in results.iterrows():
            if row['pass']:
                logger.info(f'Problem {row["name"]} passed the test')
                continue

            error = row['stderr']
            logger.debug(f'Problem {row["name"]} has error: \n{error}')

            calls = 0
            query = df_source[df_source['index'] == row['index']]['question'].values[0]
            while ("%Error" in error or "%Warning" in error) and calls < self.config.get('max_calls', 3):
                logger.info('Regenerating code due to error')

                # Implement your fix_code_from_tests_failure logic here
                solution = fix_code_from_tests_failure(query, row['dut'], error, self.config)

                df = results.loc[[i], ['index', 'tb', 'name']].copy()
                df['dut'] = solution

                result = self.verify_batch(df)
                if result.empty:
                    break
                stderr= result['stderr'].values[0]
                stdout = result['stdout'].values[0]
                error =  stderr if stderr != '' else stdout
                calls += 1

                if result['pass'].values[0]:
                    results.loc[i] = result.iloc[0]
                    print(f'{results.loc[i, 'dut'] ==  result['dut'].values[0]}')
                    logger.info(f'Problem {row['name']} passed the test after {calls} calls to generate')
                    break

            if not (results.loc[i,'pass'] or results.loc[i,'stderr'] == ''):
                logger.info(f'problem {row['name']} failed the test after {calls} calls to re-generate')

        return results
def prepare_messages(question: str, file_names: Optional[list], system_prompt: Optional[str]) -> List[Dict[str, str]]:
    """
    Prepare messages for the API request.

    Args:
        question: Problem description
        system_prompt: Optional system prompt

    Returns:
        List of formatted messages
    """
    ICL = ''
    if file_names:

        for file in file_names:
            with open(file_name, 'r') as file:
                ICL += file.read()
        ICL = '\n'.join(ICL)
        ICL = ' Here are sonme usueful information to write the code :\n' + ICL + '\n'

    index = question.find('--------------------------------------------------------------\n')
    helper = question[index:]
    format_prompt = ICL + f'''Problem description:
    ======
    {question[:index]}
    ======
    Answer the questions regarding the description
    '''
    return format_prompt, helper


def process_problems(dataset: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Process multiple problems with parallel generation and batch verification."""
    date = '2024-12-11' #datetime.now().date()


    for i in range(config.get("num_passes", 1)):
        solution_path = os.path.join(
            './outputs',
            f"{config['dataset']}_{config['model_name']}_nretry{config["max_calls"]}_nlayers{config["layers"]}_{date}_{i}.json"
            # f"{config['dataset']}_{config['model_name']}_{date}_{i}.json"
        )
        config['solution_path'] = solution_path
        logger.info(f'Working on {config['solution_path']}')
        try:
            existing_results = utils.load_json(solution_path)
            existing_data = pd.DataFrame(existing_results)
            # print(existing_data.head())
            to_run = existing_data['index'].tolist()
            df = dataset[~dataset['index'].isin(to_run)]
            logger.info(f'Number of problems to run {len(df)}')
        except:
            logger.info(f'No existing file at {solution_path}. Starting fresh.')
            df = dataset.copy()
            existing_results = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=config.get("num_parallel_calls", 4)
        ) as executor:
            futures = {
                executor.submit(generate_single_solution, item, config): item
                for item in df.to_dict('records')
            }

            results = []
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(df),
                desc="Generating Solutions"
            ):
                try:
                    result = future.result()
                    existing_results.append(result)

                    # Update file with new result while preserving existing data
                    with open(solution_path, 'w') as json_file:
                        json.dump(existing_results, json_file, indent=2)

                except Exception as e:
                    logger.error(f"Generation failed: {e}")
        final_results = []
        try:
            current_results = utils.load_json(solution_path)
            current_data = pd.DataFrame(current_results)
            # Find items needing verification
            to_test = [item['index'] for item in current_results]
               # if 'pass' not in item or item['pass'] is None]
            df_to_verify = current_data[current_data['index'].isin(to_test)]

            logger.info(f'Number of problems to verify {len(df_to_verify)}')
            verifier = BatchVerifier(config)
            verified_results = verifier.verify_in_batches(df_to_verify, dataset)
            if not verified_results.empty:
                # Update existing results with verification results
                for idx, row in verified_results.iterrows():
                    result_idx = next(
                        (i for i, r in enumerate(current_results)
                        if r['index'] == row['index']),
                        None
                    )
                    if result_idx is not None:
                        current_results[result_idx].update(row.to_dict())

                # Save updated results
                with open(solution_path, 'w') as json_file:
                    json.dump(current_results, json_file, indent=2)

                final_results.extend(current_results)
        except Exception as e:
            logger.error(f"Verification failed: {e}")
    return pd.DataFrame(final_results) if final_results else pd.DataFrame()

def generate_single_solution(item: Dict, config: Dict[str, Any]) -> Dict:
    """Generate solution for a single problem."""
    item['ICL_filenames'] = None
    # prompt,_ = prepare_messages(item['query'], item['ICL_filenames'], config.get('system_prompt'))
    prompt,_ = prepare_messages(item['question'], item['ICL_filenames'], config.get('system_prompt'))
    try:
        response, history = one_problem(prompt, config)
        return {
            'index': item['index'],
            'dut': response,
            'conversation': history,
            'tb': item['tb'],
            'name': item['name']
        }
    except Exception as e:
        logger.error(f"Failed to generate solution: {e}")
        raise

def main():
    """Main entry point for the script."""
    import argparse

    parser = argparse.ArgumentParser(description="Code generation and verification framework")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = json.load(f)

        df = pd.read_pickle(config['dataset_path'])
        if config.get('selected_indices'):
            df = df.iloc[config['selected_indices']].reset_index(drop=True)

        results = process_problems(df, config)

        if not results.empty:
            logger.info("Processing completed successfully")

    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        raise

if __name__ == '__main__':
    main()
