# make_single_channel.py
# This script is used to generate single channel data from multi-channel data

import os
import ffmpeg
import argparse

def get_audio_channels(file_path):
    try:
        probe = ffmpeg.probe(file_path, select_streams='a:0', show_entries='stream=channels', of='json')
        channels = probe['streams'][0]['channels']
        return channels
    except ffmpeg.Error as e:
        print(f"Error probing {file_path}: {e}")
        return None

def convert_to_mono(input_file, output_file):
    try:
        ffmpeg.input(input_file).output(output_file, ac=1).run(overwrite_output=True)
        print(f"Converted {input_file} to {output_file}")
    except ffmpeg.Error as e:
        print(f"Error converting {input_file} to mono: {e}")

def process_files(input_directory, output_directory):
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                channels = get_audio_channels(file_path)
                
                if channels == 2:
                    relative_path = os.path.relpath(file_path, input_directory)
                    output_path = os.path.join(output_directory, relative_path)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    convert_to_mono(file_path, output_path)
                elif channels == 1:
                    print(f"Skipping mono file: {file_path}")
                else:
                    print(f"Skipping file with {channels} channels: {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dual channel audio files to single channel")
    parser.add_argument("--input_dir", help="Path to the input directory containing audio files")
    parser.add_argument("--output_dir", help="Path to the output directory to store single channel audio files")
    
    args = parser.parse_args()
    
    process_files(args.input_dir, args.output_dir)
    print(f"All dual channel audio files have been converted to single channel and saved to {args.output_dir}.")

