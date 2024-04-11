from pathlib import Path
import json, os, torch


def load_string(file_name):
    try: return Path(file_name).read_text()
    except: return ''

def save_string(string, file_name):
    try: Path(file_name).write_text(string)
    except: return print(f"Error writing to {file_name=}")

def load_json(file_name): return json.loads(load_string(file_name))

def save_json(obj, file_name): save_string(json.dumps(obj), file_name)


def save_torch(obj, file_path, *, replace=True, **params): 
    if not os.path.isfile(file_path) or replace: return torch.save(obj, file_path, **params) # wrapper to unify notation
    else:
        print(f"File {file_path} already exists ! Use replace=True to overwrite it.")