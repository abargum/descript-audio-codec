import os
import librosa
import torch
import numpy as np
import gradio as gr
import json
import torch.nn as nn
from torchfcpe import spawn_bundled_infer_model
from rave.blocks2 import SpeakerRAVE
from rave.pqmf import CachedPQMF as PQMF

# Initialize FCPE model
pitch_model = spawn_bundled_infer_model(device="cuda")

class StatsModule(nn.Module):
    def __init__(self, stats):
        super().__init__()
        for k, v in stats.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            self.register_buffer(k, v)

def load_speaker_statedict(path):
    loaded_state = torch.load(path, map_location="cuda")
        
    newdict = {}
    pqmfdict = {}
    delete_list = []
        
    for name, param in loaded_state.items():
        new_name = name.replace("__S__.", "")
            
        if "pqmf" in new_name:
            new_name = new_name.replace("pqmf.", "")
            pqmfdict[new_name] = param
        else:
            newdict[new_name] = param
                
        delete_list.append(name)
    loaded_state.update(newdict)
    for name in delete_list:
        del loaded_state[name]
                
    return loaded_state, pqmfdict

def remove_silence(audio, sr, min_silence_duration=0.5, threshold_db=-40):
    """
    Remove silent passages from audio.
    
    Parameters:
    - audio: numpy array of audio samples
    - sr: sample rate
    - min_silence_duration: minimum duration (in seconds) to consider as silence
    - threshold_db: threshold in decibels below which audio is considered silence
    
    Returns:
    - numpy array with silent passages removed
    """
    threshold_amp = librosa.db_to_amplitude(threshold_db)
    
    frame_length = 1024
    hop_length = 512
    
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    speech_frames = energy > threshold_amp
    min_silence_frames = int(min_silence_duration * sr / hop_length)
    speech_samples = np.zeros_like(audio, dtype=bool)
    
    for i in range(len(speech_frames)):
        start_sample = i * hop_length
        end_sample = min(start_sample + hop_length, len(audio))
        speech_samples[start_sample:end_sample] = speech_frames[i]
    
    extend_samples = int(0.1 * sr)  # 100ms extension on each side
    speech_samples = np.convolve(speech_samples, np.ones(extend_samples), mode='same') > 0
    non_silent_indices = np.where(speech_samples)[0]
    
    if len(non_silent_indices) == 0:
        print("Warning: No speech detected in audio")
        return audio  # Return original if no speech detected
    
    filtered_audio = audio[non_silent_indices]
    
    print(f"Removed silence: original length {len(audio)/sr:.2f}s, new length {len(filtered_audio)/sr:.2f}s")
    return filtered_audio

def extract_utterance_fcpe(y, sr: int, frame_len_samples: int):
    f0_target_length = (y.shape[-1] // frame_len_samples)
    f0 = pitch_model.infer(y.unsqueeze(-1),
                           sr=sr,
                           decoder_mode='local_argmax',
                           threshold=0.006,
                           f0_min=50,
                           f0_max=750,
                           interp_uv=False,
                           output_interp_target_length=f0_target_length)
    return f0

def get_f0_fcpe(x, fs: int, win_length: int):
    f0 = extract_utterance_fcpe(x, fs, win_length)
    return f0

def extract_f0_mean_std(f0s: torch.Tensor):
    f0s = f0s[~torch.isnan(f0s)]
    f0s = f0s[f0s > 0]
    if len(f0s) > 0:
        f0s_mean = torch.mean(f0s)
        f0s_std = torch.std(f0s)
        return f0s_mean, f0s_std
    return torch.tensor(0.0), torch.tensor(0.0)

def process_audio_chunk(audio_chunk, sr, speaker_encoder, pqmf):
    """Process a single chunk of audio"""
    audio_chunk = torch.tensor(audio_chunk).unsqueeze(0)
    
    if audio_chunk.shape[-1] < 131072:
        pad = torch.zeros(1, 131072-audio_chunk.shape[-1])
        audio_chunk = torch.cat((audio_chunk, pad), dim=-1)
    
    f0 = get_f0_fcpe(audio_chunk, sr, 2048)
    f0_mean, f0_std = extract_f0_mean_std(f0)
    
    audio_multiband = pqmf(audio_chunk.unsqueeze(0))
    emb = speaker_encoder(audio_multiband)
    
    return {
        'f0_mean': f0_mean,
        'f0_std': f0_std,
        'embedding': emb
    }

def adapt_speaker(audio_path, speaker_encoder, pqmf, output_file=None, remove_silence_params=None, skip_first_seconds=2.0):
    """
    Process a single audio file in chunks and save stats to json file
    
    Parameters:
    - audio_path: Path to the audio file
    - speaker_encoder: The speaker encoder model
    - pqmf: The PQMF filter
    - output_file: Path for output json file (optional)
    - remove_silence_params: Dict with parameters for silence removal (optional)
      e.g., {'min_silence_duration': 0.5, 'threshold_db': -40}
    - skip_first_seconds: Number of seconds to skip from the beginning
    """
    
    chunk_size = 131072
    all_results = []
    
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_file = f"{base_name}_stats.json"
    
    print(f"Processing {audio_path}...")
    
    audio, sr = librosa.load(audio_path, sr=44100, mono=True)

    skip_samples = int(skip_first_seconds * sr)
    if skip_samples > 0 and skip_samples < len(audio):
        audio = audio[skip_samples:]
        print(f"Skipped first {skip_first_seconds} seconds")
    
    if remove_silence_params is not None:
        params = remove_silence_params.copy()
        if 'min_silence_duration' not in params:
            params['min_silence_duration'] = 0.5
        if 'threshold_db' not in params:
            params['threshold_db'] = -40
        
        audio = remove_silence(audio, sr, **params)
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        
        if len(chunk) >= chunk_size // 2:
            result = process_audio_chunk(chunk, sr, speaker_encoder, pqmf)
            if not torch.isnan(result['f0_mean']) and result['f0_mean'] > 0:
                all_results.append(result)
                
    print(f"Processed {len(all_results)} chunks from audio file")
    
    if all_results:
        f0_means = torch.stack([r['f0_mean'] for r in all_results])
        f0_stds = torch.stack([r['f0_std'] for r in all_results])
        embeddings = torch.stack([r['embedding'] for r in all_results])
        
        stats = {
            'f0_mean': f0_means.mean().detach().cpu(),
            'f0_std': f0_stds.mean().detach().cpu(),
            'f0_min': f0_means.min().detach().cpu(),
            'f0_max': f0_means.max().detach().cpu(),
            'embedding': embeddings.mean(dim=0).cpu(),
            'num_chunks': torch.tensor(len(all_results))
        }

        embeddings = embeddings.squeeze(0)
        emb_mean = embeddings.mean(dim=0)

        json_ready_data = {}

        json_ready_data['imported'] = {
            "avg_emb": emb_mean.squeeze().tolist(),
            "avg_one": embeddings[0][0].tolist(),
            "f0_mean": f0_means.mean().tolist(),
            "f0_std": f0_stds.mean().tolist()
        }

        with open(output_file, 'w') as fp:
            json.dump(json_ready_data, fp)
        
        print(f"Stats saved to {output_file}")
        
        return stats, output_file
    else:
        print("No valid audio chunks were processed")
        return None, None

speaker_encoder = None
pqmf = None

def init_models():
    global speaker_encoder, pqmf
    if speaker_encoder is None:
        print("Loading speaker encoder model...")
        speaker_encoder = SpeakerRAVE()
        spk_state, pqmf_state = load_speaker_statedict("scripts/rave/model000000075.model")
        speaker_encoder.load_state_dict(spk_state)
        speaker_encoder.eval()
        
        pqmf = PQMF(attenuation=100, n_band=16)
        print("Models loaded successfully")

def process_audio(audio_file, silence_duration, silence_threshold, skip_seconds):
    try:
        init_models()
        os.makedirs("temp_outputs", exist_ok=True)
        
        possible_extensions = ['.wav', '.mp3', '.flac']
        for ex in possible_extensions:
            if ex in audio_file:
                out_name = audio_file.replace(ex, '')
        output_file = os.path.join("temp_outputs", f"{os.path.basename(out_name)}_stats.json")
        
        remove_silence_params = {
            'min_silence_duration': silence_duration,
            'threshold_db': silence_threshold
        }
        
        stats, pt_path = adapt_speaker(
            audio_file, 
            speaker_encoder, 
            pqmf, 
            output_file=output_file,
            remove_silence_params=remove_silence_params,
            skip_first_seconds=skip_seconds
        )
        
        if stats is not None:
            result_text = (
                f"Analysis Results:\n"
                f"F0 Mean: {stats['f0_mean']:.2f} Hz\n"
                f"F0 Standard Deviation: {stats['f0_std']:.2f} Hz\n"
                f"F0 Range: {stats['f0_min']:.2f} - {stats['f0_max']:.2f} Hz\n"
                f"Number of chunks processed: {stats['num_chunks']}\n"
                f"Results saved to: {pt_path}"
            )
            return result_text, pt_path
        else:
            return "No valid audio chunks were processed. Please check your audio file and parameters.", None
    
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        return f"Error processing audio: {str(e)}\n\n{error_msg}", None

# Create the Gradio interface
def create_gradio_ui():
    with gr.Blocks(title="Audio Analysis Tool") as app:
        gr.Markdown("# Speaker Audio Analysis Tool")
        gr.Markdown("Upload an audio file to analyze F0 statistics and speaker embeddings")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(type="filepath", label="Upload Audio File")
                
                with gr.Row():
                    silence_duration = gr.Slider(
                        minimum=0.1, 
                        maximum=5.0, 
                        value=2.0, 
                        step=0.1,
                        label="Minimum Silence Duration (seconds)"
                    )
                    silence_threshold = gr.Slider(
                        minimum=-60, 
                        maximum=-20, 
                        value=-40, 
                        step=1,
                        label="Silence Threshold (dB)"
                    )
                
                skip_seconds = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    value=2.0,
                    step=0.5,
                    label="Skip Initial Seconds"
                )
                
                process_button = gr.Button("Process Audio", variant="primary")
            
            with gr.Column():
                results_text = gr.Textbox(label="Analysis Results", lines=10)
                pt_file = gr.File(label="Download json File")
        
        process_button.click(
            fn=process_audio,
            inputs=[audio_input, silence_duration, silence_threshold, skip_seconds],
            outputs=[results_text, pt_file]
        )
        
        gr.Markdown("""
        ## Usage Instructions
        1. Upload an audio file (WAV, MP3, etc.)
        2. Adjust parameters if needed:
           - **Minimum Silence Duration**: How long a silence must be to be removed (in seconds)
           - **Silence Threshold**: Audio level below which is considered silence (in dB)
           - **Skip Initial Seconds**: How many seconds to skip from the beginning of the audio
        3. Click "Process Audio" to analyze
        4. View results and download the json file
        """)
    
    return app

if __name__ == "__main__":
    app = create_gradio_ui()
    app.launch(share=True)