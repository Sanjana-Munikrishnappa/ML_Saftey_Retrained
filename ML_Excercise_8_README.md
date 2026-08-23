# ML Exercise 8 — Adversarial Robustness Testing (FGSM Attack)
This notebook loads the three pre-trained models from Exercise 4 (traffic light, pedestrian, vehicle) and tests how easily each one can be fooled by small, deliberately crafted pixel changes — an adversarial attack known as the Fast Gradient Sign Method (FGSM).
## What it does
FGSM perturbs an image just enough, in the direction that most increases the model's loss, to potentially flip its prediction — without the change being obviously visible to a human. The notebook applies this attack at three strengths (epsilon = 0.01, 0.05, 0.10) to a sample image and to a 100-image evaluation subset, then measures how much each of the three models' recall drops as the attack gets stronger.
## Requirements
Google Colab, plus torch, torchvision, numpy, pandas, matplotlib, and Pillow.
## Dataset location
/content/drive/MyDrive/carla_dataset/, containing test.zip and the saved model files traffic_light_model.pth, vehicle_model.pth, and pedestrian_model.pth from Exercise 4.
## Output
**Setup and model loading:** the state dictionary keys and weight-tensor shapes for the traffic light model are printed when first inspected, confirming the architecture before the model class is defined and the weights loaded. Later, the vehicle and pedestrian models are loaded the same way, and a confirmation check prints whether all three models exist and are ready to use.

**Sample image inspection:** the first test image is displayed, and its width and height are printed. The clean model prediction is also shown as a printed probability (after sigmoid), which serves as the baseline before any attack.

**Single-image attack demonstration:** for each epsilon value (0.01, 0.05, 0.10), the notebook displays the clean image next to its adversarial version side by side, so you can see how visible the perturbation is at each strength — at 0.01 it should be barely noticeable, while 0.10 shows a much more obvious "noisy" texture. For epsilon 0.05 and 0.10, the model's predicted probability is printed for both the clean and adversarial version of the image, directly showing whether the attack flipped or shifted the prediction.

**Batch robustness evaluation:** using a fixed 100-image evaluation set, the notebook runs each of the three models (traffic light, vehicle, pedestrian) against clean and adversarial images at each epsilon value. For every model/epsilon combination, three numbers are printed: the count of clean images predicted positive, the count of adversarial images predicted positive, and the resulting "Recall Drop" — how much the model's positive-detection rate fell once the attack was applied.

**Ground-truth-based recall calculation:** after loading the actual labels for the evaluation images and matching them up by filename, the notebook prints how many of the 100 evaluation images had a matching label (and how many didn't). It then computes and prints, for each model, the clean recall (epsilon = 0.0) and the adversarial recall at each epsilon (0.01, 0.05, 0.10), followed by the recall drop for each — this time measured against true ground-truth labels rather than just raw positive counts, making it a more rigorous robustness measure.

**Final summary:** all recall results (clean recall, adversarial recall, and recall drop, for every model and every epsilon) are consolidated into a single summary table. This table is the main takeaway: it shows which of the three models (traffic light, vehicle, or pedestrian) is most vulnerable to adversarial attacks, and how quickly that vulnerability grows as epsilon increases — larger recall drops mean the model is easier to fool with small pixel perturbations.
