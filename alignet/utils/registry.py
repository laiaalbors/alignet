# utils/registry.py

class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, name):
        def decorator(obj):
            self._registry[name] = obj
            return obj
        return decorator

    def get(self, name):
        if name not in self._registry:
            raise ValueError(f"{name} is not registered!")
        return self._registry[name]

DATASET_REGISTRY = Registry()
ARCHITECTURE_REGISTRY = Registry()
MODEL_REGISTRY = Registry()
LOSS_REGISTRY = Registry()
METRIC_REGISTRY = Registry()
