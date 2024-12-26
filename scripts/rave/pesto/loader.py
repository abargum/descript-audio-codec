import os
from typing import Optional

import torch

from .data import Preprocessor
from .model import PESTO, Resnet1d

def load_model(checkpoint: str,
               load_checkpoint: bool,
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
    
    if load_checkpoint:  # handle user-provided checkpoints
        model.load_state_dict(state_dict, strict=False)
        print("You loaded a pretrained network")
    else:
        print("You loaded the default network")
        
    return model