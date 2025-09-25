# losses/__init__.py
import os
import importlib

module_dir = os.path.dirname(__file__)
for fname in os.listdir(module_dir):
    if fname.endswith("_loss.py") and fname != "__init__.py":
        importlib.import_module(f"{__name__}.{fname[:-3]}")
