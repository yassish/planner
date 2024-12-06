import gzip
import json
import pickle
import pandas as pd
from datetime import datetime
def jsonl_gz_to_pickle(jsonl_gz_path, pickle_path):
    # Read and parse the .jsonl.gz file
    data = []
    with gzip.open(jsonl_gz_path, 'rt', encoding='utf-8') as f:
        for line in f:
            # Parse each line as JSON and append to the list
            data.append(json.loads(line))
    data = pd.DataFrame(data)
    
    data.rename(columns = {'prefix':'query'}, inplace=True)
    #print(data)
    # Save the parsed data to a .pkl file
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)




if __name__ == '__main__':

    # Example usage
    date  = datetime.now().date()
    date = str(date).replace("-", "")
    jsonl_gz_path = 'tests.jsonl.gz'  # Path to your .jsonl.gz file
    pickle_path = f'df_demo_{date}.pkl'         # Path where you want to save the .pkl file

    jsonl_gz_to_pickle(jsonl_gz_path, pickle_path)