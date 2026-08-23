# ML Exercise 7 — Out-of-Distribution Detection (MSP vs. kNN)
This notebook loads the three pre-trained models from Exercise 4 (traffic light, pedestrian, vehicle) and studies how well two different techniques — Maximum Softmax Probability (MSP) and k-Nearest Neighbors (kNN) — can detect when an image comes from conditions the model wasn't trained on (fog, night, a different town) versus normal "sunny" conditions it has seen before.
## What it does
Images from four conditions (sunny, fog, night, and a different town layout) are run through the traffic light model to collect confidence scores and feature vectors. These are used to test two OOD-detection approaches: MSP, which flags OOD images as ones the model is less confident about, and kNN, which flags OOD images as ones whose internal features are unusually far from typical in-distribution examples. The same confidence analysis is repeated for the pedestrian and vehicle models. Finally, MSP and kNN are directly compared against each other on fog, night, and town data using AUROC scores.
## Requirements
Google Colab, plus torch, torchvision, pandas, numpy, matplotlib, Pillow, and scikit-learn (for NearestNeighbors and AUROC calculation).
## Dataset location
/content/drive/MyDrive/carla_dataset/, containing test.zip, test-fog.zip, test-night.zip, and test-town-01.zip, plus the saved model files traffic_light_model.pth, pedestrian_model.pth, and vehicle_model.pth from Exercise 4.
## Output
**Dataset checks:** file/folder listings confirm each zip extracted correctly, and the number of images found in each condition folder (sunny, fog, night, town) is printed after listing filenames.

**Sample image grids:** early on, 5 sample images are displayed for each condition individually (titled "Sunny," "Fog," "Night," "Town"), and later a combined 4×5 grid shows samples from all four conditions side by side for direct visual comparison.

**Model inspection:** the traffic light checkpoint's saved keys and the shape of every weight/bias tensor are printed when the model is first loaded, followed by the full printed model architecture (layer types, kernel sizes, channel counts). A single traffic light image is also traced step by step through the network, printing the tensor shape after each major layer (Conv1, Pool1, Pool2, Conv3, and the flattened vector), which shows how the image's dimensions shrink as it passes through the CNN.

**Single-prediction output:** for one sample image, the model's raw output score is printed, followed by that score converted into a 0–1 probability via sigmoid — showing exactly how a raw logit becomes a confidence value. A grid of 5 sunny images then displays each one with its predicted probability as the image title.

**Score distributions:** the get_scores function collects sigmoid confidence scores for the traffic light model across all four conditions, and the notebook prints how many scores were collected per condition, followed by four histograms (one per condition) showing how those confidence scores are distributed. In-distribution (sunny) scores typically cluster near 0 or 1, while OOD scores (fog/night/town) tend to be more spread out or shifted.

**MSP AUROC:** using the sunny scores as "normal" and the fog/night/town scores as "abnormal," an AUROC score is printed for MSP-based OOD detection — a single number between 0 and 1, where closer to 1 means the confidence score alone is a strong signal for spotting OOD images.

**kNN AUROC:** internal feature vectors are extracted from the sunny (in-distribution) images and used to build a k-nearest-neighbors reference set. Each image's distance to its nearest sunny neighbors is computed, and this distance is used the same way as the confidence score above to compute a second AUROC value — this time for the kNN-based approach.

**Confidence across models and conditions:** the average confidence of all three models (traffic light, pedestrian, vehicle) is calculated and printed for sunny, fog, and night conditions, followed by a summary printout laying out all nine average-confidence numbers (3 models × 3 conditions) in one place, making it easy to see which model's confidence drops the most in bad conditions.

**Final MSP vs. kNN comparison:** AUROC scores for both methods are printed side by side for each OOD scenario (fog, night, town), along with the absolute "gap" between them. This final comparison is the main conclusion of the notebook — it shows whether the simpler MSP method or the feature-based kNN method is a more reliable way to catch out-of-distribution input for this driving model, and whether that answer changes depending on the type of distribution shift (fog vs. night vs. new location).
