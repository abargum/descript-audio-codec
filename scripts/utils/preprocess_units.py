import os
import librosa
import numpy as np
import pickle
import torch
import torch.nn as nn
from create_kmeans import kmeans
from torchaudio.functional import resample
from transformers import AutoProcessor, WavLMModel
from Amphion.models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
from huggingface_hub import hf_hub_download

# Load HuBERT model
discrete_units = torch.hub.load("bshall/hubert:main", "hubert_discrete", trust_repo=True).to(torch.device("cuda"))
discrete_units.eval()

# Load WavLM model with pre-trained kmeans
pretrained_path = "scripts/utils/kmeans_512_wavlm.pt"
num_clusters = 512
model_wavlm = WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus").to(torch.device("cuda"))
kmean_unit_extractor = kmeans(pretrained=True, clusters=num_clusters, checkpoint=pretrained_path)

#model_name = "facebook/w2v-bert-2.0"  # Use the appropriate W2VBert model
#processor = AutoProcessor.from_pretrained(model_name)
#model = AutoModel.from_pretrained(model_name).to(device)

fa_encoder = FACodecEncoder(
    ngf=32,
    up_ratios=[2, 4, 5, 5],
    out_channels=256,
)
encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
fa_encoder.load_state_dict(torch.load(encoder_ckpt))
fa_encoder = fa_encoder.to('cuda')
fa_encoder.eval()

fa_decoder = FACodecDecoder(
    in_channels=256,
    upsample_initial_channel=1024,
    ngf=32,
    up_ratios=[5, 5, 4, 2],
    vq_num_q_c=2,
    vq_num_q_p=1,
    vq_num_q_r=3,
    vq_dim=256,
    codebook_dim=8,
    codebook_size_prosody=10,
    codebook_size_content=10,
    codebook_size_residual=10,
    use_gr_x_timbre=True,
    use_gr_residual_f0=True,
    use_gr_residual_phone=True,
)
decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")
fa_decoder.load_state_dict(torch.load(decoder_ckpt))
fa_decoder = fa_decoder.to('cuda')
fa_decoder.eval()

def get_wavlm_units(audio):
    output = model_wavlm(audio)
    output = output.last_hidden_state.squeeze(0)
    units = kmean_unit_extractor.predict(output.squeeze().detach().cpu().numpy())
    units = torch.tensor(units, dtype=torch.long)
    return units, output.squeeze().detach().cpu()

def get_hubert_units(audio):
    units = discrete_units.units(audio.unsqueeze(0))
    return units.detach().cpu()

def get_acoustic_tokens(audio):
    enc_out = fa_encoder(audio)
    vq_post_emb, vq_id, _, quantized, spk_embs = fa_decoder(enc_out, eval_vq=False, vq=True)
    residual_code = vq_id[3:]
    return residual_code.transpose(0,1)

def get_features(file_path, sr):
    """Extract feature from audio file."""
    x, sr = librosa.load(file_path, sr=sr, mono=True)
    x = torch.tensor(x).unsqueeze(0).to(torch.device('cuda'))

    #zero-pad end if x is smaller than input to network
    if x.shape[-1] < 65536:
        zeros = torch.zeros(1, 65536 - x.shape[-1]).to(torch.device('cuda'))
        x = torch.cat((x, zeros), dim=-1)

    #zero pad end with one second to ensure that the offset does not go out of range
    zeros = torch.zeros(1, sr).to(torch.device('cuda'))
    x = torch.cat((x, zeros), dim=-1)
    
    x_resampled = resample(x, sr, 16000)

    hubert_units = get_hubert_units(x_resampled)
    wavlm_units, wavlm_output = get_wavlm_units(x_resampled)

    acoustic_token = get_acoustic_tokens(x_resampled.unsqueeze(1))

    #if wavlm units a smaller repeat last value
    if wavlm_units.shape[0] < hubert_units.shape[0]:
        diff = hubert_units.shape[0] - wavlm_units.shape[0]
        last_val = wavlm_units[-1]
        wavlm_units = torch.cat([wavlm_units, last_val.repeat(diff)])
        
        last_val = wavlm_output[-1:, :]  
        wavlm_output = torch.cat([wavlm_output, last_val], dim=0)
        
    return hubert_units, wavlm_units, wavlm_output, acoustic_token

def process_audio_directory(base_dirs, output_path, sample_rate):
    """
    Process multiple directories of audio files.
    
    Args:
        base_dirs (str or list): Single directory path or list of directory paths
        output_path (str): Path to save the output pickle file
        sample_rate (int): Target sample rate for audio processing
    """
    # Convert single directory to list for consistent handling
    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]
    
    audio_data = {}
    
    # Process each base directory
    for base_dir in base_dirs:
        print(f"\nProcessing directory: {base_dir}")
        
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(('.wav', '.flac')):
                    file_path = os.path.join(root, file)
                    print(f"Processing {file_path}...")
                    
                    try:
                        hubert_units, wavlm_units, wavlm_output, acoustic_token = get_features(file_path, sample_rate)
                        
                        audio_data[file_path] = {
                            'hubert_units': hubert_units,
                            'wavlm_units': wavlm_units,
                            'wavlm_output': wavlm_output,
                            'acoustic_token': acoustic_token
                        }   
                        
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
                        continue
    
    # Save all processed data
    with open(output_path, 'wb') as f:
        pickle.dump(audio_data, f)
    print(f"\nSaved units to {output_path}")
    print(f"Processed {len(audio_data)} files in total")

# Example usage
base_directories = [
    "VCTK-Corpus/wav48",
    "validation-set",
]
sample_rate = 44100
output_file = "metadata_w_wavlm_full.pkl"

process_audio_directory(base_directories, output_file, sample_rate)