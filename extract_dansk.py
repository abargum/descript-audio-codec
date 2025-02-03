import os
import requests
from bs4 import BeautifulSoup
import zipfile
import urllib.parse
import librosa
import numpy as np
import soundfile as sf

def download_and_unzip_files(url, download_dir='.'):
    """Download and unzip files from a webpage."""
    os.makedirs(download_dir, exist_ok=True)
    
    response = requests.get(url)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    zip_links = [
        urllib.parse.urljoin(url, link.get('href')) 
        for link in soup.find_all('a') 
        if link.get('href', '').lower().endswith('.zip')
    ]
    
    for zip_url in zip_links:
        filename = os.path.basename(urllib.parse.urlparse(zip_url).path)
        local_path = os.path.join(download_dir, filename)
        
        print(f"Downloading: {zip_url}")
        zip_response = requests.get(zip_url)
        zip_response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            f.write(zip_response.content)
        
        # Unzip the file
        with zipfile.ZipFile(local_path, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
        
        print(f"Extracted: {filename}")
        
    return download_dir

def process_wav_files(input_directory, output_directory, min_length=10, segment_length=5):
    """Process WAV files by removing silence and segmenting."""
    os.makedirs(output_directory, exist_ok=True)
    
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                
                try:
                    audio, sr = librosa.load(file_path, sr=None)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    continue
                
                duration = librosa.get_duration(y=audio, sr=sr)
                if duration <= min_length:
                    output_path = os.path.join(output_directory, file)
                    sf.write(output_path, audio, sr)
                    continue
                
                non_silent = librosa.effects.split(audio, top_db=30)
                audio_no_silence = np.concatenate([audio[start:end] for start, end in non_silent])
                
                if len(audio_no_silence) / sr > segment_length:
                    segments = librosa.util.frame(
                        audio_no_silence, 
                        frame_length=int(segment_length * sr), 
                        hop_length=int(segment_length * sr)
                    )
                    
                    for i, segment in enumerate(segments.T):
                        output_filename = f"{os.path.splitext(file)[0]}_segment_{i+1}.wav"
                        output_path = os.path.join(output_directory, output_filename)
                        sf.write(output_path, segment, sr)
                else:
                    output_path = os.path.join(output_directory, file)
                    sf.write(output_path, audio_no_silence, sr)
                
                print(f"Processed: {file}")

def main(url, download_dir, output_dir):
    """Main function to download, unzip, and process files."""
    # Download and unzip
    unzipped_dir = download_and_unzip_files(url, download_dir)
    
    # Process WAV files
    process_wav_files(unzipped_dir, output_dir)

if __name__ == "__main__":
    webpage_url = "https://sprogtek-ressources.digst.govcloud.dk/nota/Inspiration%202008%20-%202016/"  # Replace with actual URL
    download_directory = "./downloads"
    output_directory = "./processed_audio"
    
    main(webpage_url, download_directory, output_directory)