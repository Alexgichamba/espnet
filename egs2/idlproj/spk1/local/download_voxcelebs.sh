#!/usr/bin/bash

# Base URL for the files to download
base_url="https://cn01.mmai.io/download/voxceleb?key=3d93ffb8abe44162fc40590cc9e0fbedd8be20e71cdea9f720c29f761d1f5b99c125eed81b36bb690d09691b4cda6f2186d7be8d72892012833d219210e70c8d696a620c8ba9fc6b27b3814170694c590dab570ff25ad2803a64ce132e79d7e6f203db26ae5c7629dc1cc58dad72592161d30e40570af7225f1ea0b824e4293d"

# Array of file identifiers to download
files=(
    "vox1_dev_wav_partaa"
    "vox1_dev_wav_partab"
    "vox1_dev_wav_partac"
    "vox1_dev_wav_partad"
    "vox1_test_wav.zip"
)

# Loop over the file identifiers to download them one by one
for file_id in "${files[@]}"
do
    echo "Downloading ${file_id}..."
    wget -c "${base_url}&file=${file_id}" -O "${file_id}"
done

# Concatenate the downloaded files into a single zip file
echo "Concatenating all parts into vox1_dev_wav.zip..."
cat vox1_dev* > vox1_dev_wav.zip

echo "Download and concatenation complete!"
