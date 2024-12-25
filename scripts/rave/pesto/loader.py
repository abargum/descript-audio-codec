import os
from typing import Optional

import torch

from .data import Preprocessor
from .model import PESTO, Resnet1d

def load_model(checkpoint: str,
               step_size: float,
               sampling_rate: Optional[int] = None,
               **hcqt_kwargs) -> PESTO:
    r"""Load a trained model from a checkpoint file.
    See https://github.com/SonyCSLParis/pesto-full/blob/master/src/models/pesto.py for the structure of the checkpoint.

    Args:
        checkpoint (str): path to the checkpoint or name of the checkpoint file (if using a provided checkpoint)
        step_size (float): hop size in milliseconds
        sampling_rate (int, optional): sampling rate of the audios.
            If not provided, it can be inferred dynamically as well.
    Returns:
        PESTO: instance of PESTO model
    """
    if os.path.exists(checkpoint):  # handle user-provided checkpoints
        model_path = checkpoint
        # load checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        hparams = checkpoint["hparams"]
        state_dict = checkpoint["state_dict"]
        hcqt_params = checkpoint["hcqt_params"]
        hcqt_params.update(hcqt_kwargs)
    
        # instantiate preprocessor
        preprocessor = Preprocessor(hop_size=step_size, sampling_rate=sampling_rate, **hcqt_params)
    
        # instantiate PESTO encoder
        encoder = Resnet1d(**hparams["encoder"])
    
        # instantiate main PESTO module and load its weights
        model = PESTO(encoder,
                      preprocessor=preprocessor,
                      crop_kwargs=hparams["pitch_shift"],
                      reduction=hparams["reduction"])
        
        model.load_state_dict(state_dict, strict=False)
        print("You loaded a pretrained network")
    else:
        hparams = {'encoder': {'n_chan_input': 1,
                               'n_chan_layers': [40, 30, 30, 10, 3],
                               'n_prefilt_layers': 2,
                               'prefilt_kernel_size': 39,
                               'residual': False,
                               'n_bins_in': 147,
                               'output_dim': 384,
                               'activation_fn': 'leaky',
                               'a_lrelu': 0.3,
                               'p_dropout': 0.2,
                               'fc_margin': 0,
                               'final_norm': 'softmax'},
                   
                   'pitch_shift': {'min_steps': -16,
                                   'max_steps': 16},
                                   'reduction': 'alwa'}
        
        hcqt_params = {'harmonics': [1], 'fmin': 55.0, 'fmax': None, 'bins_per_semitone': 3, 'n_bins': 179, 'center_bins': True,
                       'gamma': 5, 'streaming': False, 'max_batch_size': 32, 'mirror': 1.0}

        preprocessor = Preprocessor(hop_size=step_size, sampling_rate=sampling_rate, **hcqt_params)
        encoder = Resnet1d(**hparams["encoder"])
        model = PESTO(encoder,
                      preprocessor=preprocessor,
                      crop_kwargs=hparams["pitch_shift"],
                      reduction=hparams["reduction"])

        print("You loaded the default network")
        
    return model