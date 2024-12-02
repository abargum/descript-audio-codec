from rave.rave_model import RAVE
import cached_conv as cc
import librosa
import torch
from scipy.io import wavfile 

from rave.pitch import get_f0_fcpe, extract_f0_mean_std
from rave.pitchTracker import PitchRegisterTracker
from rave.yin import YIN

targets = {'p228' : {'mean': 188.81, 'std': 42.20, 'gender': 'female', 'path': 'scripts/rave/audio/p228_test.flac'},
           'p225' : {'mean': 172.42, 'std': 42.39, 'gender': 'female', 'path': 'scripts/rave/audio/p225_test.flac'},
           'p227' : {'mean': 118.73, 'std': 27.41, 'gender': 'male',   'path': 'scripts/rave/audio/p227_test.flac'},
           'p226' : {'mean': 111.21, 'std': 23.87, 'gender': 'male',   'path': 'scripts/rave/audio/p226_test.flac'},
           'p238' : {'mean': 167.48, 'std': 52.83, 'gender': 'female', 'path': 'audio/output/test_manual.wav'}}


def get_manual_pitch(f0_in, tar, sr=44100):
    f0_in = f0_in[:, :, 0]
    in_med, in_std = extract_f0_mean_std(f0_in)
        
    tar_med = targets[tar]['mean']
    tar_std = targets[tar]['std']

    f0_in[f0_in == 0] = float('nan')
    standardized_source_pitch = (f0_in - in_med.to(f0_in)) / in_std.to(f0_in)
    source_pitch = (standardized_source_pitch * torch.tensor(tar_std).to(f0_in)) + torch.tensor(tar_med).to(f0_in)
    source_pitch[torch.isnan(source_pitch)] = 0
    return source_pitch

if __name__ == "__main__":

    cc.use_cached_conv(True)

    generator = RAVE()

    kwargs = {
            "folder": f"pretrained/snake/",
            "map_location": "cuda",
            "package": False,
        }

    generator, g_extra = generator.load_from_folder(**kwargs)
    generator.eval()

    target = 'p238'
    
    t, sr = librosa.load(targets[target]['path'], sr=44100, mono=True)
    t = torch.tensor(t[:131072]).unsqueeze(0).unsqueeze(0)

    x, sr = librosa.load("audio/p238_test.flac", sr=44100, mono=True)
    x = torch.tensor(x[:131072]).unsqueeze(0).unsqueeze(0)

    f0 = get_f0_fcpe(x.squeeze(1), sr, 1024)
    print("Retrieved pitch of size:", f0.shape)

    print("Calculating manual pitch conversion...")
    
    pitch = get_manual_pitch(f0, target)
    y = generator.predict_no_pitch(x, t, pitch)
    y = y[0, 0, :].detach().cpu().numpy()
    wavfile.write('audio/output/test_manual.wav', sr, y)

    print("Calculating tracked pitch conversion...")
    
    p_tracker = PitchRegisterTracker(target_mean = targets[target]['mean'],
                                     target_std = targets[target]['std'],
                                     buffer_size = sr // 10)

    pitch = p_tracker(f0.transpose(2, 1)).squeeze(1)
    y = generator.predict_no_pitch(x, t, pitch)
    y = y[0, 0, :].detach().cpu().numpy()
    wavfile.write('audio/output/test_tracker.wav', sr, y)