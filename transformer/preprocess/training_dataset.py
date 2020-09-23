import os
import numpy as np
import compress_pickle

from transformer.utils import get_tqdm


def create_training_dataset(english_tokens,
                            german_tokens,
                            max_samples=None,
                            sample_length=100,
                            save_dataset=False,
                            save_interval=100000,
                            save_dir=None,
                            save_compression='zipfile',
                            tqdm=None):
    max_samples = min(max_samples or len(english_tokens), len(english_tokens), len(german_tokens))
    if tqdm is None:
        tqdm = get_tqdm()
    x = []
    y = []
    y_target_position_info = []

    for sample_index, e_sample, d_sample in zip(tqdm(range(max_samples), desc='create_training_data'),
                                                english_tokens,
                                                german_tokens):
        tokens_in_sample = min(len(e_sample), len(d_sample), sample_length)

        e_sample_entries = _preprocess_create_training_data_samples(e_sample, sample_length, tokens_in_sample)
        d_sample_entries = _preprocess_create_training_data_samples(d_sample, sample_length, tokens_in_sample)
        y_sample_target_position = np.arange(0, tokens_in_sample)

        x.append(e_sample_entries)
        y.append(d_sample_entries)
        y_target_position_info.append(y_sample_target_position)

        if sample_index % save_interval == 0 and save_dataset:
            os.makedirs(save_dir, exist_ok=True)

            x = np.concatenate(x)
            y = np.concatenate(y)
            y_target_position_info = np.concatenate(y_target_position_info)
            assert len(x) == len(y) == len(y_target_position_info), \
                f'unexpected data mismatch for x={len(x)}, y={len(y)}, y_target={len(y_target_position_info)}'

            dataset = (x, y, y_target_position_info)
            compress_pickle.dump(dataset,
                                 path=os.path.join(save_dir, f'train_{sample_index}'),
                                 compression=save_compression)
            x = []
            y = []
            y_target_position_info = []

    if save_dataset:
        return

    x = np.concatenate(x)
    y = np.concatenate(y)
    y_target_position_info = np.concatenate(y_target_position_info)

    assert len(x) == len(y) == len(y_target_position_info), \
        f'unexpected data mismatch for x={len(x)}, y={len(y)}, y_target={len(y_target_position_info)}'

    return x, y, y_target_position_info


def _preprocess_create_training_data_samples(sample,
                                             sample_length,
                                             tokens_in_sample):
    if sample_length < len(sample):
        sample = sample[:sample_length]

    sample_entries = np.tile(sample, reps=(tokens_in_sample, 1))
    sample_entries = np.tril(sample_entries)

    padding_length = sample_length - len(sample)
    if padding_length:
        sample_entries = np.pad(sample_entries, ((0, 0), (0, padding_length)), mode='constant', constant_values=0)
    return sample_entries
