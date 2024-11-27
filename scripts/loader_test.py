from rave.rave_model import RAVE
import cached_conv as cc
import librosa
import torch
from scipy.io import wavfile 

if __name__ == "__main__":

    cc.use_cached_conv(True)

    generator = RAVE()

    kwargs = {
            "folder": f"pretrained/rave-gen/",
            "map_location": "cuda",
            "package": False,
        }

    generator, g_extra = generator.load_from_folder(**kwargs)

    generator.eval()

    t, sr = librosa.load("scripts/rave/p228_005_mic1.flac", sr=44100, mono=True)
    t = torch.tensor(t[:131072]).unsqueeze(0).unsqueeze(0)

    x, sr = librosa.load("audio/reklame.mp3", sr=44100, mono=True)
    x = torch.tensor(x[:2*131072]).unsqueeze(0).unsqueeze(0)

    y = generator.predict(x, t)

    y = y[0, 0, :].detach().cpu().numpy()
    wavfile.write('test_reklame.wav', sr, y)

    