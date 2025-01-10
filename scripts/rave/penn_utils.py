import torch

###############################################################################
# Audio parameters
###############################################################################


# Width of a pitch bin
CENTS_PER_BIN = 5.  # cents

# Whether to trade quantization error for noise during inference
DITHER = False

# Minimum representable frequency
FMIN = 31.  # Hz

# Distance between adjacent frames
HOPSIZE = 80  # samples

# The size of the window used for locally normal pitch decoding
LOCAL_PITCH_WINDOW_SIZE = 19

# Pitch velocity constraint for viterbi decoding
MAX_OCTAVES_PER_SECOND = 35.92

# Whether to normalize input audio to mean zero and variance one
NORMALIZE_INPUT = False

# Number of spectrogram frequency bins
NUM_FFT = 1024

# One octave in cents
OCTAVE = 1200  # cents

# Number of pitch bins to predict
PITCH_BINS = 1440

# Audio sample rate
SAMPLE_RATE = 44100  # hz

# Size of the analysis window
WINDOW_SIZE = 1024  # samples

GAUSSIAN_BLUR = True

LOSS = 'categorical_cross_entropy'


###############################################################################
# Pitch conversions
###############################################################################


def bins_to_cents(bins):
    """Converts pitch bins to cents"""
    return CENTS_PER_BIN * bins


def bins_to_frequency(bins):
    """Converts pitch bins to frequency in Hz"""
    return cents_to_frequency(bins_to_cents(bins))


def cents_to_bins(cents, quantize_fn=torch.floor):
    """Converts cents to pitch bins"""
    bins = quantize_fn(cents / CENTS_PER_BIN).long()
    bins[bins < 0] = 0
    bins[bins >= PITCH_BINS] = PITCH_BINS - 1
    return bins


def cents_to_frequency(cents):
    """Converts cents to frequency in Hz"""
    return FMIN * 2 ** (cents / OCTAVE)


def frequency_to_bins(frequency, quantize_fn=torch.floor):
    """Convert frequency in Hz to pitch bins"""
    return cents_to_bins(frequency_to_cents(frequency), quantize_fn)


def frequency_to_cents(frequency):
    """Convert frequency in Hz to cents"""
    return OCTAVE * torch.log2(frequency / FMIN)


def frequency_to_samples(frequency, sample_rate=SAMPLE_RATE):
    """Convert frequency in Hz to number of samples per period"""
    return sample_rate / frequency


def frequency_to_midi(frequency):
    """
    Convert frequency to MIDI note number(s)
    Based on librosa.hz_to_midi(frequencies) implementation
    https://librosa.org/doc/main/_modules/librosa/core/convert.html#hz_to_midi
    """
    return 12 * (torch.log2(frequency) - torch.log2(torch.tensor(440.0))) + 69


def midi_to_frequency(midi):
    """
    Convert MIDI note number to frequency
    Based on librosa.midi_to_hz(notes) implementation
    https://librosa.org/doc/main/_modules/librosa/core/convert.html#midi_to_hz
    """
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


###############################################################################
# Time conversions
###############################################################################


def frames_to_samples(frames):
    """Convert number of frames to samples"""
    return frames * HOPSIZE


def frames_to_seconds(frames):
    """Convert number of frames to seconds"""
    return frames * HOPSIZE_SECONDS


def seconds_to_frames(seconds):
    """Convert seconds to number of frames"""
    return samples_to_frames(seconds_to_samples(seconds))


def seconds_to_samples(seconds, sample_rate=SAMPLE_RATE):
    """Convert seconds to number of samples"""
    return seconds * sample_rate


def samples_to_frames(samples):
    """Convert samples to number of frames"""
    return samples // HOPSIZE


def samples_to_seconds(samples, sample_rate=SAMPLE_RATE):
    """Convert number of samples to seconds"""
    return samples / sample_rate