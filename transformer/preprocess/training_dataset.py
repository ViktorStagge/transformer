import os
import numpy as np
import compress_pickle

from transformer.utils import get_tqdm
from transformer.utils import get_logger

logger = get_logger('transformer')


def create_training_dataset(english_tokens,
                            german_tokens,
                            max_samples=None,
                            sample_length=100,
                            save_dataset=True,
                            save_interval=1000,
                            save_dir=None,
                            save_compression='zipfile',
                            tqdm=None):
    max_samples = min(max_samples or len(english_tokens), len(english_tokens), len(german_tokens))
    # save_interval = min(save_interval, len(english_tokens), len(german_tokens))

    if tqdm is None:
        tqdm = get_tqdm()

    assert len(english_tokens) == len(german_tokens), \
        f'unexpected data mismatch for english={len(english_tokens)}, german={len(german_tokens)}'

    if save_dataset:
        os.makedirs(save_dir, exist_ok=True)

        logger.debug(f'creating dataset: i={0}, max_samples={max_samples}, step={save_interval}')
        for i in tqdm(range(0, max_samples, save_interval), desc='create_training_data'):
            logger.debug(f'creating dataset: i={i}, max_samples={max_samples}, step={save_interval}')
            assert isinstance(english_tokens, list), \
                f'unexpected format received: received={type(english_tokens)}, expected=<list>'

            en = english_tokens[i: i+save_interval]
            en = [row[:sample_length] for row in en]

            de = german_tokens[i: i+save_interval]
            de = [row[:sample_length] for row in de]

            assert len(en) == len(de), \
                f'unexpected data mismatch for english={len(en)}, german={len(de)}'

            dataset = (en, de)
            compress_pickle.dump(dataset,
                                 path=os.path.join(save_dir, f'train_{i}'),
                                 compression=save_compression)

    if not save_dataset:
        raise NotImplementedError('Down-prioritized due to too typically high memory requirements.')
        assert len(en) == len(de), \
            f'unexpected data mismatch for english={len(en)}, german={len(de)}'

        return en, de
