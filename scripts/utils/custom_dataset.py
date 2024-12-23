import torch
import math
from typing import Union
from flatten_dict import flatten
from flatten_dict import unflatten
from audiotools import AudioSignal
from audiotools.data.datasets import AudioDataset

class CustomAudioDataset(AudioDataset):
    @staticmethod
    def collate(list_of_dicts: list, n_splits: int = None):
    
        batches = []
        list_len = len(list_of_dicts)
    
        return_list = False if n_splits is None else True
        n_splits = 1 if n_splits is None else n_splits
        n_items = int(math.ceil(list_len / n_splits))
    
        for i in range(0, list_len, n_items):
            # Flatten the dictionaries to avoid recursion.
            list_of_dicts_ = [flatten(d) for d in list_of_dicts[i : i + n_items]]
            dict_of_lists = {
                k: [dic[k] for dic in list_of_dicts_] for k in list_of_dicts_[0]
            }
    
            batch = {}
            offsets = []
            for k, v in dict_of_lists.items():
                if isinstance(v, list):
                    if all(isinstance(s, AudioSignal) for s in v):
                        for s in v:
                            offsets.append(s.metadata['offset'])
                        batch[k] = AudioSignal.batch(v, pad_signals=True)
                        batch[k].metadata['offset'] = offsets
                    else:
                        # Borrow the default collate fn from torch.
                        batch[k] = torch.utils.data._utils.collate.default_collate(v)
            batches.append(unflatten(batch))
    
        batches = batches[0] if not return_list else batches
        return batches