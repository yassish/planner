from torch.utils.data import Dataset


def getFinalSystemPrompt(system_prompt, results):
    """Construct a system prompt for layers 2+ that includes the previous responses to synthesize."""
    return (
        system_prompt
        + "\n"
        + "\n".join([f"{i+1}. {str(element)}" for i, element in enumerate(results)])
    )


class Preprocess(Dataset):
    """
    You can add any kind of preprocessing here, such as few shot prompts or role playing and others.
    """
    def __init__(self, raw_dataset, config):
        self.raw_dataset = raw_dataset
        self.config = config

    def __getitem__(self, ix: int):
        # add preprocessing here if needed
        # pre_prompt = """A python docstring with the initial definition of a function is given to. You need to write the best function that represents the prompt request. Use '```python' and '```' to mark the beginning and end of the code. ```python """
        raw_prompt = self.raw_dataset[ix]['prompt']
        item = {
            'pre_prompt': self.config['pre_prompt'],
            'raw_prompt': raw_prompt,
            'test': self.raw_dataset[ix]['test'],
            'task_id' : self.raw_dataset[ix]['task_id'],
            'func_name': self.raw_dataset[ix]['entry_point'],
            'canonical_solution': self.raw_dataset[ix]['canonical_solution'],
            'index': ix
        }
        return item

    def __len__(self) -> int:
        return len(self.raw_dataset)
    
import json
def load_json(json_path):
    import json
    """Load JSON data from a file.

    Args:
        json_path (str): The path to the JSON file.

    Returns:
        dict: Parsed JSON data.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def load_yaml(response_text: str, keys_fix_yaml: List[str] = []) -> dict:
    """
    Load and parse YAML data from a given response text.

    Args:
        response_text (str): The text containing YAML data.
        keys_fix_yaml (List[str]): A list of keys to fix in the YAML if necessary. Defaults to an empty list.

    Returns:
        dict: The parsed YAML data as a dictionary.

    This function attempts to parse the provided YAML text and returns the resulting dictionary. If parsing fails, it attempts to fix the YAML format.
    """
    response_text = response_text.rstrip("` \n")
    response_text = response_text.removeprefix('```yaml').rstrip('`')
    try:
        data = yaml.safe_load(response_text)
    except Exception as e:
        data = try_fix_yaml(response_text, keys_fix_yaml=keys_fix_yaml)
        if not data:
            get_logger().info(f"Failed to parse AI YAML prediction: {e}")
    return data


def try_fix_yaml(response_text: str, keys_fix_yaml: List[str] = []) -> dict:
    """
    Attempt to fix and parse YAML data from a given response text.

    Args:
        response_text (str): The text containing YAML data.
        keys_fix_yaml (List[str]): A list of keys to fix in the YAML if necessary. Defaults to an empty list.

    Returns:
        dict: The successfully parsed YAML data as a dictionary.

    Raises:
        ValueError: If YAML parsing fails after attempting to fix it.

    This function modifies the YAML text to fix common formatting issues and attempts to parse it. If successful, it returns the parsed data; otherwise, it raises an error.
    """
    response_text_lines = response_text.split('\n')

    keys = keys_fix_yaml
    response_text_lines_copy = response_text_lines.copy()
    for i in range(0, len(response_text_lines_copy)):
        for key in keys:
            if response_text_lines_copy[i].strip().startswith(key) and not '|' in response_text_lines_copy[i]:
                response_text_lines_copy[i] = response_text_lines_copy[i].replace(f'{key}',
                                                                                  f'{key} |-\n        ')
    try:
        data = yaml.safe_load('\n'.join(response_text_lines_copy))
        get_logger().info(f"Successfully parsed AI prediction after adding |-\n")
        return data
    except Exception as e:
        logger.debug('\n'.join(response_text_lines_copy))

