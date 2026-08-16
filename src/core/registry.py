from typing import Type, Dict, Callable

class Registry:
    """
    Generic Registry for components
    """

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Type] = {}

    
    def register(self, name: str) -> Callable:
        """
        Decorator to register class with name string
        """

        def wrapper(class_object: Type) -> Type:
            if name in self._registry:
                raise ValueError(f"{name} already registered in {self._name}")
            
            self._registry[name] = class_object
            return class_object

        return wrapper
    
    
    def get(self, name: str) -> Type:
        if name not in self._registry:
            available = list(self._registry.keys())

            raise KeyError(f"{name} not found in {self._name}. "f"Available: {available}")

        return self._registry[name]
    

    def list_registered(self) -> list:
        return list(self._registry.keys())
    

# Registeries for diff component types:
DOWNSAMPLER_REGISTRY = Registry("DOWNSAMPLER_REGISTRY")
BACKBONE_REGISTRY = Registry("BACKBONE_REGISTRY")
LOSS_REGISTRY = Registry("LOSS_REGISTRY")
