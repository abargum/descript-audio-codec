import os
import shutil
import random

def copy_random_audio_files(source_dir, destination_dir, num_files=1000):
    # Create the destination directory if it does not exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Collect all .wav and .flac files from the source directory (including subdirectories)
    audio_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(('.wav', '.flac')):
                audio_files.append(os.path.join(root, file))

    # Check if there are enough files to copy
    if len(audio_files) < num_files:
        print(f"Only {len(audio_files)} audio files found. Copying all of them.")
        num_files = len(audio_files)

    # Randomly select the files to copy
    selected_files = random.sample(audio_files, num_files)

    # Copy the selected files to the destination directory
    for file in selected_files:
        shutil.copy(file, destination_dir)

    print(f"Copied {len(selected_files)} files to '{destination_dir}'.")

# Example usage
source_directory = "VCTK-Corpus/wav48"  # Replace with the source directory
destination_directory = "validation-set"  # Replace with the destination directory
copy_random_audio_files(source_directory, destination_directory, 1000)
