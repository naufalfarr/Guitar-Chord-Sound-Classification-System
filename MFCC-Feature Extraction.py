import librosa
import os
import json
import matplotlib.pyplot as plt

num_mfcc = 40 
n_fft = 2048
hop_length = 512

# Define your dataset path and output JSON path
DATASET_PATH = r"F:\File Kuliah\Semester 5\Pengenalan Pola\Projek\Training"
JSON_PATH = r"hasilAm.json"
SAMPLES_TO_CONSIDER = 44100  # 1 sec. of audio

def preprocess_dataset(dataset_path, json_path, num_mfcc=40, n_fft=2048, hop_length=512):
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # Create a dictionary to map folder names to labels
    label_mapping = {}

    # Get a list of subdirectories in the dataset_path
    subdirs = os.listdir(dataset_path)

    for i, subdir in enumerate(subdirs):
        label_mapping[subdir] = i

    for subdir in subdirs:
        subdir_path = os.path.join(dataset_path, subdir)

        if os.path.isdir(subdir_path):
            data["mapping"].append(subdir)
            label = label_mapping[subdir]

            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)

                # Load audio file and slice it to ensure length consistency among different files
                signal, sample_rate = librosa.load(file_path)

                if len(signal) >= SAMPLES_TO_CONSIDER:
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # Extract MFCCs
                    MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40, n_fft=2048, hop_length=512)

                    # Store data for analyzed track
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["labels"].append(label)
                    data["files"].append(file_path)

    # Save data in a JSON file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH)

