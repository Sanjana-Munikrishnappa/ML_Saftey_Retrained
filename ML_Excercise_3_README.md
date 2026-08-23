# ML Exercise 3 — Traffic Light, Pedestrian & Vehicle Detection  

This notebook trains three image classifiers on the CARLA driving-simulator dataset, each answering a yes/no question about a photo: does it contain a traffic light, a pedestrian, or a vehicle? All three use the same code, just retrained on a different label.

## How it works
The dataset is stored in Google Drive as zipped train, validation, and test folders, which the notebook mounts and extracts. Each split also has a labels.feather file loaded into a pandas DataFrame, with one row per image and columns like has_traffic_light, has_pedestrian, and has_vehicle.

A custom CarlaDataset class loads each image, resizes it to 224×224, and pairs it with its label from the DataFrame. A DataLoader then batches and shuffles this data for training.

The model, TrafficLightClassifier, is a small CNN: three convolution + pooling blocks followed by fully connected layers that output one confidence score. It's trained for 3 epochs using BCEWithLogitsLoss and the Adam optimizer, printing the average loss after each epoch.

After training, the model is evaluated on the validation set — predictions are converted from raw scores to 0/1 using a sigmoid and 0.5 cutoff, then compared to the true labels to get a validation accuracy percentage. This whole process (data setup, training, validation) repeats for the pedestrian and vehicle models, and a loss curve is plotted for each.

## Requirements
Google Colab, plus torch, torchvision, pandas, pyarrow, matplotlib, and Pillow.

## Dataset location
/content/drive/MyDrive/carla_dataset/ containing train.zip, validation.zip, and test.zip.

## How to run

Open in Colab, adjust dataset_path if needed, and run all cells top to bottom.

## Output
For each of the three models: per-epoch training loss, a final validation accuracy, and a loss-curve plot. Weights stay in memory only — they aren't saved to disk.
