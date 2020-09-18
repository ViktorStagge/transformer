import sys
import logging

from typing import Union, \
                   Iterable, \
                   Callable

_loggers = {}


def get_tqdm(method: Union[str, Iterable] = None,
             **kwargs) -> Callable:
    """Retrieves the regular tqdm progress-bar or the tqdm-notebook progress-bar.

    Passing an already created tqdm progress-bar will return the same reference as is.

    Arguments:
        method: type of progress-bar to use

    Returns:
        tqdm: an iterable progress bar
    """
    try:
        from tqdm import tqdm, \
                         tqdm_notebook
    except ImportError as e:
        return mock_tqdm

    if method is False or method in ['iter', 'mock']:
        return mock_tqdm
    if method is None or method in ['tqdm']:
        return tqdm
    if method in ['notebook', 'tqdm-notebook']:
        return tqdm_notebook
    return tqdm


def mock_tqdm(iterable=None, **kwargs):
    for value in iter(iterable):
        yield value


def get_logger(name: str = None,
               format: str = None,
               datefmt: str = None):
    global _loggers

    if format is None:
        format = '|%(asctime)s, %(name)s| %(message)s'
    if datefmt is None:
        datefmt = '%Y-%m-%d %H:%M'

    if name not in _loggers:
        logger = Logger(name=name)

        formatter = logging.Formatter(format, datefmt=datefmt)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        _loggers[name] = logger

    return _loggers[name]


class Logger(logging.Logger):
    """Logger instance representing one logging channel.
    Instance is callable, intended as a conveniency method
    for the `Logger.info(..)` method.
    """

    def __call__(self, message, *args, **kwargs):
        self.info(message, *args, **kwargs)
