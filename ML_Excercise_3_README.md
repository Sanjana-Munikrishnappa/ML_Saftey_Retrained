# ML Exercise 3 — Traffic Light, Pedestrian & Vehicle Detection

This notebook trains simple CNN classifiers on the CARLA driving-simulator dataset to detect:

🚦 Traffic lights
🚶 Pedestrians
🚗 Vehicles

## What it does
Loads the data – Mounts Google Drive, unzips the train/validation/test datasets, and loads labels from Feather files.
Explores the data – Checks class balance and previews sample images.
Builds a Dataset/DataLoader – A custom PyTorch CarlaDataset class loads images and matching labels.
Trains 3 models – One TrafficLightClassifier CNN is trained separately for each task (traffic light, pedestrian, vehicle detection).
Evaluates each model – Checks accuracy on the validation set.
Plots results – Shows training loss curves for each model.

## Requirements
Google Colab (uses Google Drive mount)
Python packages: torch, pandas, pyarrow (for Feather files), matplotlib, PIL

## Dataset
Expects zipped CARLA dataset files in Google Drive:

/content/drive/MyDrive/carla_dataset/
├── train.zip
├── validation.zip
└── test.zip

## How to run

Open the notebook in Google Colab.
Update dataset_path if your dataset is stored elsewhere.
Run all cells top to bottom.
Check the accuracy printouts and loss plots at the end.

## Output
Trained weights for 3 classifiers (in memory — not saved to disk by default)
Validation accuracy for each model
Loss curve plots for each model
