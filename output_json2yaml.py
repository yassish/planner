import yaml
import json
import sys
import os

def yaml_multiline_string_presenter(dumper, data):
    if len(data.splitlines()) > 1:
        # Pyyaml does not allow trailing space at the end of line for block string
        data = '\n'.join([line.rstrip() for line in data.strip().splitlines()])
        # Pyyaml does not allow tab in a block string
        data = data.replace('\t', '    ')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')

    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

yaml.add_representer(str, yaml_multiline_string_presenter)
yaml.representer.SafeRepresenter.add_representer(str, yaml_multiline_string_presenter)


if __name__ == '__main__':

    assert len(sys.argv) > 1, f'Need 1 arg, the input json filepath'
    json_fpath = sys.argv[1]
    assert json_fpath
    assert os.path.exists(json_fpath)

    with open(json_fpath) as f:
        data = json.load(f)

    # deal with conversation - [NEW STEP]
    for x in data:
        l = x.get('conversation', str()).split('[NEW STEP]')
        x['conversation'] = {key: str(i) for key, i in enumerate(l)}

    print( yaml.dump(data, allow_unicode=True) )
