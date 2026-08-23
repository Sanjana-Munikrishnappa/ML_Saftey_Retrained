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
Running the notebook produces output at several stages, mostly printed to the console or shown as inline plots — nothing is saved to disk unless you add that yourself.

**During data exploration:** the number of samples in each split (train/validation/test) is printed, along with the column names in the labels table. Bar charts show the positive vs. negative count for each label (has_traffic_light, has_pedestrian, has_vehicle), and a row of sample images is displayed from the training folder.

**During training (repeated once per model):** each of the 3 epochs prints one line showing the average loss for that epoch, for example Epoch 1/3, Loss: 0.3618. Watching these three numbers drop across epochs is how you tell the model is learning — a loss that stays flat or increases means something's off with training.

**During validation (repeated once per model):** a single line is printed with the model's accuracy on unseen validation data, for example Validation Accuracy: 91.42%. This is calculated by running every validation image through the trained model, converting its output into a 0-or-1 prediction, and checking what fraction match the true labels.

**At the end:** a loss-curve plot is shown for each of the three models — epoch number on the x-axis, loss on the y-axis. A smoothly descending line means healthy training; a jagged or flat one suggests the model struggled.

**What you're left with:** three trained models (traffic light, pedestrian, vehicle), each with its own loss numbers, one validation accuracy score, and one loss plot — but only in the Colab session's memory. If you want to keep a model beyond that session, you'd need to add code to save its state_dict to a file.
