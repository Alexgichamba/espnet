#!/usr/bin/env python3
import os
import argparse
import soundfile as sf
import numpy as np
from multiprocessing import Pool

def process_wav_file(args):
    """Process a single WAV file and return its shape information"""
    utt_id, wav_path, sample_rate = args
    if sample_rate:
        sample_rate = int(sample_rate.replace('k', '000'))
    try:
        # Read audio file
        audio, file_sample_rate = sf.read(wav_path)
        
        # Resample if specific sample rate is requested
        if sample_rate and file_sample_rate != sample_rate:
            raise ValueError(f"Sample rate mismatch: file sample rate of {file_sample_rate} != fs argument {sample_rate}")
        
        return f"{utt_id} {audio.shape[0]}"
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None

def generate_speech_shapes(wav_scp_path: str, output_dir: str, nj: int = 1, sample_rate: int = None):
    """
    Generate shape files from wav.scp file using multiprocessing
    
    Args:
        wav_scp_path (str): Path to wav.scp file
        output_dir (str): Output directory for shape files
        nj (int): Number of parallel processes
        sample_rate (int, optional): Desired sample rate for resampling
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read wav.scp file
    with open(wav_scp_path, 'r') as wav_scp:
        wav_entries = [
            line.strip().split() for line in wav_scp 
        ]
    
    # Prepare arguments for multiprocessing
    process_args = [(utt_id, wav_path, sample_rate) for utt_id, wav_path in wav_entries]
    
    # Process files in parallel
    with Pool(processes=nj) as pool:
        shapes = pool.map(process_wav_file, process_args)
    
    # Write shape file
    shape_file_path = os.path.join(output_dir, 'speech_shape')
    with open(shape_file_path, 'w') as shape_file:
        for shape in filter(None, shapes):
            shape_file.write(f"{shape}\n")

def main():
    parser = argparse.ArgumentParser(description='Generate speech shape files')
    parser.add_argument('--wav_scp', help='Path to wav.scp file', required=True)
    parser.add_argument('--output_dir', help='Output directory for shape files', required=True)
    parser.add_argument('--fs', type=str, help='Desired sample rate (optional)', default=None)
    parser.add_argument('--nj', type=int, help='Number of parallel jobs', default=1)
    
    args = parser.parse_args()
    
    generate_speech_shapes(args.wav_scp, args.output_dir, args.nj, args.fs)

if __name__ == '__main__':
    main()