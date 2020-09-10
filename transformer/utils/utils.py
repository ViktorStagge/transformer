from typing import Union, \
                   Iterable


def get_tqdm(method: Union[str, Iterable] = None,
             **kwargs) \
        -> Iterable:
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
