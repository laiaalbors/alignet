# datasets/__init__.py
# Auto-import all *_dataset.py modules so their @DATASET_REGISTRY.register() decorators run.
import os
import importlib

module_dir = os.path.dirname(__file__)
for fname in sorted(os.listdir(module_dir)):
    if fname.endswith("_dataset.py"):
        importlib.import_module(f"{__name__}.{fname[:-3]}")
