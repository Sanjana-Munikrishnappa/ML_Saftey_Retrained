#  ML Exercise 9 — Model Calibration for Object Detection
This notebook loads the three pre-trained models from Exercise 4 (traffic light, vehicle, pedestrian) and measures how well-calibrated their confidence scores are — that is, whether a "70% confident" prediction is actually right about 70% of the time. It then applies temperature scaling to improve calibration and analyzes the real-world cost trade-offs of different confidence thresholds.
## What it does
Expected Calibration Error (ECE) is computed for each model on the test set, followed by reliability diagrams that visualize calibration quality. Temperature scaling is then applied — searching for the temperature value that best calibrates each model on a validation set — and ECE is recalculated afterward to see the improvement. Finally, a custom cost function (combining false negatives and false positives) is used to compare the pedestrian model's performance across different calibration states and classification thresholds.
## Requirements
Google Colab, plus torch, torchvision, numpy, pandas, and matplotlib.
## Dataset location
/content/drive/MyDrive/carla_dataset/, containing train.zip, validation.zip, and test.zip, plus the saved model files traffic_light_model.pth, vehicle_model.pth, and pedestrian_model.pth from Exercise 4.
## Output
**Setup and label checks:** directory listings confirm each dataset archive extracted correctly. The labels table is previewed (first few rows and column names), and a count of positive examples for each target (traffic light, pedestrian, vehicle) is printed to show class balance in the test set.

**Model loading:** each model's checkpoint keys are printed when first loaded, confirming the saved layers before the model classes are defined and weights applied.

**Uncalibrated ECE:** for each of the three models, a single ECE value is printed — a number between 0 and 1 where lower is better. This measures the gap between predicted confidence and actual correctness, averaged across confidence bins, before any correction is applied.

**Reliability diagrams:** for each model, a plot is shown with predicted confidence on one axis and actual accuracy on the other, broken into bins. A perfectly calibrated model would trace the diagonal line; bars that dip below or rise above the diagonal reveal over-confidence or under-confidence at that confidence range.

**Temperature scaling search:** for each model, the notebook tests a range of temperature values against the validation set and prints the Negative Log-Likelihood (NLL) for each, then reports the single best temperature value that minimizes NLL — this is the calibration "fix" being solved for.

**Post-calibration ECE:** each model's ECE is recalculated using its newly found optimal temperature and printed again, so it can be directly compared to the earlier uncalibrated number — a lower post-calibration ECE means temperature scaling improved how trustworthy the model's confidence scores are.

**ECE comparison table:** a summary table (as a pandas DataFrame) lists all three models with their ECE before and after calibration side by side, making the improvement from temperature scaling easy to see across the board.

**Cost analysis for the pedestrian model:** using a custom cost function that penalizes false negatives and false positives, four scenarios are evaluated and printed — uncalibrated at threshold 0.5, uncalibrated at a very low threshold (0.0099), calibrated at threshold 0.5, and calibrated at the low threshold. Each printout reports the false negative count, false positive count, and total cost for that configuration.

**Cost comparison table:** all four cost scenarios are consolidated into one summary table, and the notebook explicitly identifies and prints which configuration (calibrated or not, and which threshold) produced the lowest overall cost. This final result is the practical payoff of the whole notebook — it shows that calibration and threshold choice aren't just academic; they directly affect how many pedestrians the model misses or falsely flags, which is a genuine safety trade-off for a system like this.
