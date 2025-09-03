import importlib.util
import yaml
from types import ModuleType
from typing import Any

def load_from_py(file_path: str, var_name: str) -> Any:
    spec = importlib.util.spec_from_file_location("config_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    
    if not hasattr(module, var_name):
        raise AttributeError(f"Variable '{var_name}' not found in {file_path}")
    
    return getattr(module, var_name)

def load_from_yaml(file_path:str) -> dict:
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.UnsafeLoader)
    return data