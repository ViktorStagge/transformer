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
                            validation_split=None,
                            save_dataset=True,
                            save_interval=1000,
                            save_dir=None,
                            save_dir_validation=None,
                            save_compression='zipfile',
                            tqdm=None):
    max_samples = min(max_samples or len(english_tokens), len(english_tokens), len(german_tokens))
    # save_interval = min(save_interval, len(english_tokens), len(german_tokens))

    if tqdm is None:
        tqdm = get_tqdm()

    assert len(english_tokens) == len(german_tokens), \
        f'unexpected data mismatch for english={len(english_tokens)}, german={len(german_tokens)}'
    assert isinstance(english_tokens, list), \
        f'unexpected format received: received={type(english_tokens)}, expected=<list>'

    if save_dataset:
        os.makedirs(save_dir, exist_ok=True)
        if save_dir_validation is not None:
            os.makedirs(save_dir_validation, exist_ok=True)
            assert validation_split is not None, \
                f'provide a validation split (fraction of whole dataset created) when using `save_dir_validation=True`'
            validation_index = int((1 - validation_split)*max_samples)
        else:
            validation_index = max_samples

        for i in tqdm(range(0, max_samples, save_interval), desc='create_training_data'):
            logger.debug(f'creating dataset: i={i}, max_samples={max_samples}, step={save_interval}')

            en = english_tokens[i: i+save_interval]
            de = german_tokens[i: i+save_interval]

            for j in range(min(len(en), len(de))):
                row_sample_length = min(len(en[j]), len(de[j]), sample_length)
                en[j] = en[j][:row_sample_length]  # Truncating to same length as a first solution
                de[j] = de[j][:row_sample_length]  # Truncating to same length as a first solution

            assert len(en) == len(de), \
                f'unexpected data mismatch for english={len(en)}, german={len(de)}'

            dataset = (en, de)
            compress_pickle.dump(dataset,
                                 path=os.path.join(save_dir if i < validation_index else save_dir_validation,
                                                   f'train_{i}'),
                                 compression=save_compression)

    if not save_dataset:
        raise NotImplementedError('Down-prioritized due to too typically high memory requirements.')
        assert len(en) == len(de), \
            f'unexpected data mismatch for english={len(en)}, german={len(de)}'

        return en, de
