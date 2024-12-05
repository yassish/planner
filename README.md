# SystemVerilog Code Generator Framework

A Python-based framework for generating and verifying SystemVerilog code using Language Models (LLM). Supports parallel code generation and automated verification through remote Verilator testing.

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
```json
{
  "num_parallel_calls": 5,          // Parallel generation threads
  "max_calls": 5,                   // Max LLM regeneration attempts on failure
  "verilator_endpoint": "http://your-verilator-endpoint:port/test_runner_api",
  "batch_size": 16,                 // Verification batch size
  "dataset_path": "./your_dataset.pkl",
  "model_name": "your_model_name",  
  // ... other configuration options
}
```

## Usage

Run the code generator:
```bash
python main.py --config path/to/config.json
```

Results are saved in the `outputs` directory and logs in the `logs` directory.

## Important Notes

1. **Remote Verification**: Code verification is performed on a remote machine specified by `verilator_endpoint` in the config file

2. **Automated Error Recovery**: The system will automatically attempt to fix failed verifications up to `max_calls` times using Verilator feedback. Each attempt generates a new solution using the LLM with the error feedback

3. **No User Feedback**: Currently, the system operates fully automatically without user feedback in the generation/verification loop. User feedback integration could be a future enhancement

## License
