# data_prep.py

import os
import pandas as pd
import argparse

def import_idmt_traffic_dataset(fn_txt: str) -> pd.DataFrame:
    """ Import IDMT-Traffic dataset
    Args:
        fn_txt (str): Text file with all WAV files
    Returns:
        df_dataset (pd.DataFrame): File-wise metadata
            Columns:
                'file': WAV filename,
                'is_background': True if recording contains background noise (no vehicle), False else
                'date_time': Recording time (YYYY-MM-DD-HH-mm)
                'location': Recording location
                'speed_kmh': Speed limit at recording site (km/h), UNK if unknown,
                'sample_pos': Sample position (centered) within the original audio recording,
                'daytime': M(orning) or (A)fternoon,
                'weather': (D)ry or (W)et road condition,
                'vehicle': (B)us, (C)ar, (M)otorcycle, or (T)ruck,
                'source_direction': Source direction of passing vehicle: from (L)eft or from (R)ight,
                'microphone': (SE)= (high-quality) sE8 microphones, (ME) = (low-quality) MEMS microphones (ICS-43434),
                'channel': Original stereo pair channel (12) or (34)
    """
    # load file list
    df_files = pd.read_csv(fn_txt, names=('file',))
    fn_file_list = df_files['file'].to_list()

    # load metadata from file names
    df_dataset = []

    for f, fn in enumerate(fn_file_list):
        fn = fn.replace('.wav', '')
        parts = fn.split('_')

        # background noise files
        if '-BG' in fn:
            date_time, location, speed_kmh, sample_pos, mic, channel = parts
            vehicle, source_direction, weather, daytime = 'None', 'None', 'None', 'None'
            is_background = True

        # files with vehicle passings
        else:
            date_time, location, speed_kmh, sample_pos, daytime, weather, vehicle_direction, mic, channel = parts
            vehicle, source_direction = vehicle_direction
            is_background = False

        channel = channel.replace('-BG', '')
        speed_kmh = speed_kmh.replace('unknownKmh', 'UNK')
        speed_kmh = speed_kmh.replace('Kmh', '')

        df_dataset.append({'file': fn,
                           'is_background': is_background,
                           'date_time': date_time,
                           'location': location,
                           'speed_kmh': speed_kmh,
                           'sample_pos': sample_pos,
                           'daytime': daytime,
                           'weather': weather,
                           'vehicle': vehicle,
                           'source_direction': source_direction,
                           'microphone': mic,
                           'channel': channel})

    df_dataset = pd.DataFrame(df_dataset, columns=('file', 'is_background', 'date_time', 'location', 'speed_kmh', 'sample_pos', 'daytime', 'weather', 'vehicle',
                                                   'source_direction', 'microphone', 'channel'))

    return df_dataset


def create_kaldi_files(metadata_df: pd.DataFrame, audio_dir: str, trg_dir: str):
    """ Create Kaldi-style wav.scp and utt2spk files.
    Args:
        metadata_df (pd.DataFrame): Metadata DataFrame
        audio_dir (str): Directory where audio files are stored
        trg_dir (str): Target directory where wav.scp and utt2spk files will be saved
    """
    wav_scp_path = os.path.join(trg_dir, 'wav.scp')
    utt2spk_path = os.path.join(trg_dir, 'utt2spk')

    os.makedirs(trg_dir, exist_ok=True)

    with open(wav_scp_path, 'w') as wav_scp_file, open(utt2spk_path, 'w') as utt2spk_file:
        for index, row in metadata_df.iterrows():
            utt_id = row['file']
            spk_id = row['vehicle']
            # Skip background noise or any invalid data
            if row['is_background'] or spk_id == 'None':
                continue
            
            wav_file_path = os.path.join(audio_dir, f"{utt_id}.wav")
            # make spk_id as prefix of utt_id
            utt_id = f"{spk_id}_{utt_id}"
            wav_scp_file.write(f"{utt_id} {wav_file_path}\n")
            utt2spk_file.write(f"{utt_id} {spk_id}\n")


def main():
    parser = argparse.ArgumentParser(description="Create Kaldi-style wav.scp and utt2spk files.")
    parser.add_argument('--audio_dir', type=str, help='Path to the directory containing audio files')
    parser.add_argument('--trg_dir', type=str, help='Path to the target directory for wav.scp and utt2spk files')
    parser.add_argument('--metadata_file', type=str, help='Path to the metadata file (e.g., idmt_traffic_all.txt)')
    
    args = parser.parse_args()
    
    metadata_df = import_idmt_traffic_dataset(args.metadata_file)
    create_kaldi_files(metadata_df, args.audio_dir, args.trg_dir)

if __name__ == '__main__':
    main()
