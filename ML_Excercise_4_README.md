# ML Exercise 4 — Traffic Light, Pedestrian & Vehicle Detection.
This notebook builds on Exercise 3: it trains the same three image classifiers on the CARLA dataset (traffic light, pedestrian, vehicle detection), but adds proper evaluation metrics and saves the trained models to Google Drive.

## How it works
The dataset is mounted from Google Drive, and the zipped train/validation/test folders are extracted. Labels for each split are loaded from CSV files into pandas DataFrames. A custom DrivingDataset class loads each image, resizes it, converts it to a tensor, and pairs it with the correct label column (has_traffic_light, has_pedestrian, or has_vehicle). DataLoaders then batch and shuffle this data for training and testing.

The model, TrafficLightClassifier, is the same small CNN as before: three convolution + pooling blocks followed by fully connected layers producing one confidence score. It trains using BCEWithLogitsLoss and the Adam optimizer, with training looping through the training DataLoader, computing loss, and updating weights each epoch.

The key difference from Exercise 3 is in evaluation. Instead of just reporting accuracy, the notebook runs the trained model on the test set and converts its raw scores to 0/1 predictions using a sigmoid and 0.5 cutoff. It then computes precision, recall, and F1-score using scikit-learn, and plots a confusion matrix to show true positives, true negatives, false positives, and false negatives. This whole process — training, loss plotting, and evaluation — repeats for the pedestrian and vehicle models.

Finally, each trained model's weights are saved to Google Drive as a .pth file using torch.save(), and the notebook lists all saved .pth files at the end to confirm they were written successfully.
## Requirements
Google Colab, plus torch, torchvision, pandas, matplotlib, Pillow, and scikit-learn.
## Dataset location
/content/drive/MyDrive/carla_dataset/ containing train.zip, validation.zip, and test.zip.
## How to run
Open in Colab, adjust the dataset path if needed, and run all cells top to bottom. Training, evaluation, and saving happen automatically for all three models.
## Output
Running the notebook produces output at several stages, mostly printed to the console or shown as inline plots and charts.

**During training (once per model):** each epoch prints its average loss, for example Epoch 1/3, Loss: 0.3618, so you can watch the loss decrease as the model learns. A loss-curve plot is shown after training finishes for each model.

**During evaluation (once per model):** three numbers are printed — precision, recall, and F1-score — measuring how well the model identifies the target class on unseen test images. A confusion matrix plot is also displayed, breaking down correct and incorrect predictions into true/false positives and negatives, which is more informative than accuracy alone, especially if the classes are imbalanced.

**During saving:** each model's weights are written to Google Drive as a .pth file (traffic_light_model.pth, pedestrian_model.pth, vehicle_model.pth), with a confirmation message printed after each save. At the end, the notebook lists every .pth file found in the save directory to verify all three were saved correctly.

**What you're left with:** three trained models with their loss curves, precision/recall/F1 scores, and confusion matrices — and unlike Exercise 3, these models persist on Google Drive as .pth files, so they can be reloaded later without retraining.
