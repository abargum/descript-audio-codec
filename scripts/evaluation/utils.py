import random
import numpy as np
import torch
import sys
import os
import os
import pickle
import torch
import torch.nn as nn

# Add root directory to path for imports
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from modules.model import VoiceModel

def set_seed(seed):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Set seed for CPU
    torch.cuda.manual_seed(seed)  # Set seed for CUDA
    torch.cuda.manual_seed_all(seed)  # Set seed for all CUDA devices
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior on GPU
    torch.backends.cudnn.benchmark = False  # Disable optimizations for non-deterministic algorithms
    print(f"Random seed {seed} has been set for reproducibility.")

def load_voice_model(model_path, device="cuda"):
    """
    Load a voice conversion model from the specified path.
    
    Args:
        model_path (str): Path to the model folder
        device (str): Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        VoiceModel: Loaded and initialized voice model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    generator = VoiceModel()
    
    kwargs = {
        "folder": model_path,
        "map_location": device,
        "package": False
    }
    
    try:
        generator, g_extra = generator.load_from_folder(**kwargs)
        generator.to(device)
        generator.eval()
        return generator
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")

def get_speaker_embeddings(targets):
    """
    Load speaker embeddings and F0 statistics for target speakers.
    
    Args:
        targets (list): List of target speaker IDs
    
    Returns:
        tuple: (embeddings, f0_means, f0_stds)
    """
    file_path = 'scripts/utils/speaker_emb_dict.pkl'
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Speaker embedding dictionary not found at: {file_path}")
    
    with open(file_path, 'rb') as file:
        speaker_dict = pickle.load(file)
    
    emb_list = nn.ParameterList()
    f0_mean_list = []
    f0_std_list = []
    
    for speaker in targets:
        if speaker not in speaker_dict:
            raise KeyError(f"Speaker '{speaker}' not found in speaker dictionary. "
                          f"Available speakers: {list(speaker_dict.keys())}")
            
        target_stats = speaker_dict[speaker]
        target_emb = torch.tensor(target_stats['avg_emb']).unsqueeze(0)
        target_f0_mean = target_stats['f0_mean']
        target_f0_std = target_stats['f0_std']
    
        emb_list.append(nn.Parameter(target_emb))
        f0_mean_list.append(target_f0_mean)
        f0_std_list.append(target_f0_std)
    
    return emb_list, f0_mean_list, f0_std_list