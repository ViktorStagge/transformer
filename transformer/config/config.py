from omegaconf import OmegaConf

from transformer.config.default import config


_configs = OmegaConf.create(dict(default=config))


def get_config(path: str = 'default',
               **kwargs):
    if path in _configs:
        return _configs[path]

    config = OmegaConf.load(path)
    _configs[path] = config
    return config
