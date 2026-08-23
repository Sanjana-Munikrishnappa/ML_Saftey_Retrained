# ML Safety – CARLA Perception Models

This project was submitted by Sanjana Munikrishnappa  **Matriculation number**: 261187 for the **Course**: Data and Knowledge Engineering 
 at **University**: Otto von Guericke University.



This project trains and stress-tests CNN classifiers on CARLA driving-simulator images to detect traffic lights, pedestrians, and vehicles. Across the exercises, the same three models are trained, evaluated, attacked, calibrated, and explained to study their real-world safety behavior.

The perception system consists of three binary classifiers:

1. 🚶 Pedestrians
2. 🚗 Vehicles
3. 🚦 Traffic Lights

All notebooks are built for Google Colab and use a dataset stored in Google Drive.

## Dataset
Stored in Google Drive at MyDrive/carla_dataset/:

- train.zip, validation.zip, test.zip — main dataset splits
- test-fog.zip, test-night.zip, test-town-01.zip — extra test sets under different conditions (used for robustness testing)
- traffic_light_model.pth, pedestrian_model.pth, vehicle_model.pth — saved trained models

Each split folder contains:

- rgb-front/ — front camera images (.jpg)
- labels.feather or labels.csv — labels: has_traffic_light, has_pedestrian, has_vehicle

## Requirements
pip install torch torchvision pandas pyarrow pillow matplotlib scikit-learn seaborn

Runs on Google Colab (uses google.colab.drive to mount Drive).

### How to run

1. Open any notebook in Colab.
2. Put the dataset zips (and .pth model files, once you've trained them) in Drive at the path above.
3. Run all cells top to bottom. Later notebooks (5, 6, 7, 8, 9) reuse the models trained in Exercise 3/4, so run those first. 

## Exercises
### Exercise 3 — Baseline Model Training
Trains 3 simple CNNs from scratch (traffic light, pedestrian, vehicle) on 224×224 images, 3 epochs each.
### Validation Accuracy:
| Model | Accuracy |
|---|---|
| Traffic Light | 95.94% |
| Pedestrian | 73.47% |
| Vehicle | 74.83% |

## Exercise 4 — Retraining with Proper Evaluation
Rebuilds the same 3 models with a cleaner pipeline and adds precision/recall/F1 and confusion matrix evaluation instead of just accuracy.
### Test Set Results:
| Model | Precision | Recall | F1 |
|---|---|---|---|
| Traffic Light | 0.92 | 0.97 | 0.94 |
| Pedestrian | 0.22 | 0.10 | 0.13 |
| Vehicle | 0.83 | 0.91 | 0.87 |

_Pedestrian detection is weak — likely due to class imbalance in the training data._

## Exercise 5 — Safety Analysis: Temperature Scaling & Backdoor Attack
Exercise 5 — Safety Analysis: Temperature Scaling & Backdoor Attack

Two conceptual/practical safety exercises:

- **Temperature scaling:** Shows how scaling logits (T = 0.5, 1.0, 2.0) changes model confidence, and why overconfident (low-T) models are dangerous for a safety rule like "reduce speed if confidence < 0.6."

- **Backdoor attack:** Poisons 10% of pedestrian-positive training images with a small red-square trigger and flips their label to "no pedestrian," simulating a data-poisoning attack. Evaluates the backdoored model with Clean Recall (normal test set) vs Attack Success Rate (triggered images).

## Exercise 6 — Explainability & Out-of-Domain Accuracy

- Generates saliency maps for correctly classified and misclassified images (traffic light, pedestrian, vehicle) to see which pixels the model focuses on.
- Tests the traffic-light model on fog, night, and a different town to check generalization.

### Traffic Light Accuracy by Condition:
| Condition | Accuracy |
|---|---|
| Fog | 76.72% |
| Night | 27.53% |
| Different Town | 70.56% |

_The model fails badly at night — a key real-world safety gap._

## Exercise 7 — Out-of-Distribution (OOD) Detection
Compares two methods for detecting when input images are out-of-distribution (fog/night/different town vs normal "sunny" data):

- MSP (Maximum Softmax Probability)
- kNN (k-Nearest Neighbors on features)

### AUROC (higher = better at catching OOD inputs):
| Condition | MSP AUROC | kNN AUROC |
|---|---|---|
| Fog | 0.71 | 1.00 |
| Night | 1.00 | 1.00 |
| Different Town | 0.74 | 1.00 |
_kNN is a much more reliable OOD detector than MSP here._

## Exercise 8 — Adversarial Robustness (FGSM Attack)
Attacks all 3 models using **Fast Gradient Sign Method (FGSM)** at increasing perturbation strengths (ε = 0.01, 0.05, 0.10) and measures how much recall drops.
### Recall Drop (out of sample count) by epsilon:

| Model | ε=0.01 | ε=0.05 | ε=0.10 |
|---|---|---|---|
| Traffic Light | 48 / 65 | 64 / 65 | 64 / 65 |
| Vehicle | 31 / 100 | 97 / 100 | 98 / 100 |
| Pedestrian | 3 / 4 | 4 / 4 | 4 / 4 |

_Even very small pixel perturbations (ε=0.05) collapse recall to near zero for all 3 models — none of them are adversarially robust._

## Exercise 9 — Model Calibration
Measures how well each model's confidence matches its actual accuracy using **Expected Calibration Error (ECE)**, then applies **temperature scaling** to try to fix it. Also computes a custom cost function (false negatives + false positives) for the pedestrian model at different confidence thresholds.

### ECE Before vs After Calibration:
| Model | ECE Before | ECE After |
|---|---|---|
| Traffic Light | 0.235 | 0.235 |
| Vehicle | 0.108 | 0.036 |
| Pedestrian | 0.479 | 0.493 |

_Calibration helped the vehicle model but not traffic light or pedestrian — for pedestrian, lowering the decision threshold (τ = 0.0099) gave a much lower overall cost than the default 0.5 threshold._

## Model Architecture (used across all exercises)

```
Input: 3×224×224 RGB image
Conv(3→16) → ReLU → MaxPool
Conv(16→32) → ReLU → MaxPool
Conv(32→64) → ReLU → MaxPool
Flatten → Linear(128) → ReLU → Linear(1)
```

- Loss: `BCEWithLogitsLoss`
- Optimizer: Adam (lr = 1e-3)
- 3 epochs, batch size 32

## Key Takeaways

- The traffic-light model performs best; pedestrian detection is consistently the weakest across every exercise (imbalance-driven).
- All three models fail badly under adversarial attack and struggle in unseen conditions like night driving.
- kNN-based OOD detection outperforms simple confidence-based (MSP) detection.
- Calibration and threshold tuning can meaningfully reduce real-world cost, especially for the pedestrian model.

## Repo Structure

| File | Description |
|---|---|
| `ML_Exercise_3.ipynb` | Baseline training |
| `ML_Excercise_4.ipynb` | Retraining + precision/recall/F1 |
| `ML_Excercise_5.ipynb` | Temperature scaling + backdoor attack |
| `ML_Excercise__6.ipynb` | Saliency maps + fog/night/town accuracy |
| `ML_Excercise__7.ipynb` | Out-of-distribution detection (MSP vs kNN) |
| `ML_Excercise_8.ipynb` | FGSM adversarial attack |
| `ML_Excercise_9.ipynb` | Calibration (ECE, temperature scaling, cost analysis) |
