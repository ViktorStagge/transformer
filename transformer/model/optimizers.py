from typing import Union
from keras import optimizers
from omegaconf import OmegaConf
from dataclasses import dataclass


@dataclass
class Adam:
    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.98  # 0.999 in Compressive Transformer
    clipnorm: float = 0.1


def get_optimizer(method: Union[str, optimizers.Optimizer],
                  **kwargs):
    """Retrieves the specified optimizer.

    Defaults to `Adam` with the parameters as specified by Rae et. al for their Compressive Transformer.
    As this is expected to work well for the Transformer as well.
    """
    if isinstance(method, str):
        method = method.lower()

    if method in ['adam']:
        config = OmegaConf.structured(Adam)
        config.update(**kwargs)
        return optimizers.Adam(**config)
    return method
