import torch
import torch.nn as nn
from torch import Tensor

from typing import Dict, Iterable, Callable

# code adapted from https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904



def module_hook(module: nn.Module, input: Tensor, output: Tensor):
    """
        module hook abstraction
        For nn.Module objects only.
    """
    raise NotImplementedError()
    
def tensor_hook(grad: Tensor):
    """
        module hook abstraction
        For nn.Module objects only.

        For Tensor objects only.
        Only executed during the *backward* pass!
    """

    raise NotImplementedError()

class VerboseExecution(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Register a hook for each layer
        for name, layer in self.model.named_children():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda layer, _, output: print(f"{layer.__name__}: {output.shape}")
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self.handles = {}
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            handle = layer.register_forward_hook(self.save_outputs_hook(layer_id))

            self.handles[layer_id] = handle

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def release(self):
        """
            release every registered hook.
        """
        for layer_id, handle in self.handles.items():
            handle.remove()

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features