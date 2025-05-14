import os
import shutil
import random
import argparse

def move_random_audio_files(source_dir, destination_dir, num_files=1000):
    """
    Move a random selection of audio files from source_dir to destination_dir.
    
    Args:
        source_dir (str): Source directory containing audio files
        destination_dir (str): Destination directory to move files to
        num_files (int): Number of files to move (default: 1000)
    """
    # Create the destination directory if it does not exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print(f"Created destination directory: '{destination_dir}'")
    
    # Collect all .wav and .flac files from the source directory (including subdirectories)
    audio_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(('.wav', '.flac')):
                audio_files.append(os.path.join(root, file))
    
    # Check if there are enough files to move
    if len(audio_files) < num_files:
        print(f"Only {len(audio_files)} audio files found. Moving all of them.")
        num_files = len(audio_files)
    
    # Randomly select the files to move
    selected_files = random.sample(audio_files, num_files)
    
    # Move the selected files to the destination directory
    moved_count = 0
    for file in selected_files:
        dest_path = os.path.join(destination_dir, os.path.basename(file))
        try:
            shutil.move(file, dest_path)
            moved_count += 1
        except Exception as e:
            print(f"Error moving file {file}: {e}")
    
    print(f"Moved {moved_count} files to '{destination_dir}'.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Move a random selection of audio files from source directory to destination directory.')
    
    parser.add_argument('--source', 
                        required=True,
                        help='Source directory containing audio files')
    
    parser.add_argument('--destination', 
                        required=True,
                        help='Destination directory to move files to')
    
    parser.add_argument('--count', 
                        type=int, 
                        default=1000,
                        help='Number of files to move (default: 1000)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the function with parsed arguments
    move_random_audio_files(args.source, args.destination, args.count)

if __name__ == "__main__":
    main()