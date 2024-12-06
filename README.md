# SystemVerilog Code Generator Framework

A Python-based framework for generating and verifying SystemVerilog code using Language Models (LLM). Supports parallel code generation and automated verification through remote Verilator testing.

## Overview

This tool processes SystemVerilog queries stored in a DataFrame with the following required schema:
- *name*: Name of the design
- *index*: Unique identifier
- *query*: SystemVerilog design specification
- *tb*: Testbench code
- *canonical_dut*: Reference design implementation

The framework allows customization of the LLM's reasoning process through configurable prompts in the config file. You can either use step-by-step prompting (by configuring `layer_prompt`) or direct generation (by leaving `layer_prompt` and `pre_prompt` empty).

## Setup

1. Clone the repository:
```bash
git clone git@github.com:yassish/planner.git
cd planner
```

2. Create conda environment:
```bash
conda env create -f environment.yml
conda activate planner
```

3. Create necessary directories:
```bash
mkdir logs outputs
```

## Required Files

1. `keys.json` - Contains your API keys:
```json
{
    "unified_api_key": "your-api-key"
}
```

2. `config.json` - Configuration file:
## Configuration Details

The `config.json` file controls the behavior of the code generator. Here's a complete example with descriptions:

```json
{
    "selected_indices": false,        // Whether to process specific indices from dataset. If false, processes all
    "num_passes": 10,                // Number of generation attempts per query
    "max_tokens": 6000,              // Maximum tokens per LLM response
    "temperature": 0.9,              // LLM temperature for generation (higher = more creative)
    "num_parallel_calls": 5,         // Number of parallel generation threads
    "max_calls": 0,                  // Maximum retry attempts on verification failure (0 = no retries)
    "verilator_endpoint": "http://34.223.52.212:5005/test_runner_api",  // Remote Verilator server endpoint
    "batch_size": 16,               // Number of designs to verify in parallel
    "dataset": "demo",              // Dataset identifier
    "dataset_path": "./df_demo_20241113.pkl",  // Path to dataset file
    "layers": 8,                    // Number of reasoning layers
    "seed": 42,                     // Random seed for reproducibility
    "model_name": "google_claude_3.5_sonnet_v2",  // LLM model identifier
    
    // Step-by-step prompts for guided generation
    "layer_prompts": [
        "What is the main functionality of this hardware module? What is the high-level block diagram?",
        "What aspects should be parameterizable? What are reasonable default values?",
        "How will data be processed and transformed between input and output? What registers and combinational logic are needed?",
        "What are the clocking, latency and throughput requirements? Are there specific timing constraints? What signals need to be reset? Should reset be synchronous or asynchronous?",
        "What test scenarios are needed? How will assertions be used to catch issues?",
        "What distinct functional blocks or submodules would logically divide this design? For each submodule identified, what specific task does it perform?",
        "Write a System verilog code for each of the submodules specified. What are the critical interfaces between these submodules?"
    ],
    
    // Final integration prompt
    "pre_prompt": "Integrate the submodules into one working module in SystemVerilog. Include all submodules in your final answer as one working module. Leave no comment referencing previous answers."
}
```

### Key Configuration Parameters

1. **Generation Control**
   - `max_tokens`: Limits the length of LLM responses
   - `temperature`: Controls randomness in generation (0.0-1.0)
   - `num_passes`: Number of attempts per query
   - `seed`: Ensures reproducible results

2. **Parallelization Settings**
   - `num_parallel_calls`: Controls parallel generation threads
   - `batch_size`: Number of designs verified simultaneously
   - `max_calls`: Retry attempts for failed verifications

3. **Dataset Configuration**
   - `dataset`: Identifier for the dataset being processed
   - `dataset_path`: Location of the dataset file.
   - `selected_indices`: Option to process specific dataset entries

4. **Model and Server Settings**
   - `model_name`: Specifies the LLM model to use
   - `verilator_endpoint`: Remote verification server address

5. **Generation Process**
   - `layers`: Number of reasoning steps
   - `layer_prompts`: Step-by-step questions guiding the generation
   - `pre_prompt`: Final integration instruction

> Note: Each layer prompt is designed to guide the LLM through a specific aspect of hardware design, from high-level functionality to detailed implementation.
## Results of having retry in CoT 

![Performance for each m odule](images/image.png)
## Usage

Run the code generator:
```bash
python main.py --config path/to/config.json
```

Results are saved in the `outputs` directory and logs in the `logs` directory.

## Important Notes

1. **Remote Verification**: Code verification is performed on a remote machine specified by `verilator_endpoint` in the config file.

2. **Automated Error Recovery**: The system will automatically attempt to fix failed verifications up to `max_calls` times using Verilator feedback. Each attempt generates a new solution using the LLM with the error feedback.

3. **No User Feedback**: Currently, the system operates fully automatically without user feedback in the generation/verification loop. User feedback integration could be a future enhancement.

4. **Chain of thought**: Review the layer prompts and make sure it is in order you desire. There is a last layer prompt hard-coded in the main.py, where it asks the model to write a DUT based on the previous steps if specified.

## License

