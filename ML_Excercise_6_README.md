# ML Exercise 6 — Model Interpretability & Robustness Testing
This notebook loads the three pre-trained classifiers from Exercise 4 (traffic light, pedestrian, vehicle) and probes them in two ways: it visualizes why each model makes its predictions using saliency maps, and it tests how well the traffic light model holds up when the driving conditions change (fog, night, a different town layout).
## How it works
Loads the three pre-trained models from Exercise 4 (traffic light, pedestrian, vehicle), generates saliency maps to visualize what each model focuses on when making a prediction, and tests the traffic light model's accuracy under fog, night, and an alternate town layout to check how well it generalizes beyond its training conditions.
## Requirements
Google Colab, plus torch, torchvision, pandas, numpy, matplotlib, and Pillow.
## Dataset location
/content/drive/MyDrive/carla_dataset/, containing test.zip plus the additional condition datasets test-fog.zip, test-night.zip, and test-town-01.zip, along with the saved model files traffic_light_model.pth, pedestrian_model.pth, and vehicle_model.pth from Exercise 4.
## How to run
Open in Colab, mount Drive, and run all cells top to bottom. Since this notebook loads pre-trained models rather than training new ones, it runs much faster than Exercises 3 and 4.
## Output
Running the notebook produces mostly visual output (image pairs and heatmaps) plus a handful of printed accuracy numbers, spread across several stages.

**Setup and sanity checks:** dataset lengths (e.g. number of test samples) are printed after each dataset is built, along with directory listings confirming the test, fog, night, and town folders and their rgb-front image subfolders extracted correctly. A list of .pth files found in Drive is printed to confirm all three trained models are available before loading.

**Saliency maps for correctly classified samples:** for each of the three models (traffic light, pedestrian, vehicle), the notebook displays 5 side-by-side image pairs — the original photo on the left, and a "hot" colormap heatmap on the right showing which pixels most influenced that correct prediction. Brighter/warmer areas in the heatmap mean the model paid more attention there.

**Saliency maps for misclassified samples:** for the traffic light model specifically, 3 more image pairs are shown the same way, but for images the model got wrong. Comparing these to the correct-prediction heatmaps is useful for spotting whether the model was focusing on irrelevant parts of the image (like the road or sky) instead of the actual traffic light when it made a mistake.

**Saliency maps under different conditions:** one more image-pair is displayed for each of the fog, night, and town datasets, showing how the model's attention shifts when the visual conditions differ from training — for example, whether it still locks onto the traffic light in low-light or foggy scenes, or gets distracted by noise in the image.

**Final accuracy comparison:** three printed lines report the traffic light model's accuracy on each out-of-distribution dataset, for example Fog Accuracy: 78.50%, Night Accuracy: 65.20%, Town Accuracy: 82.10% (actual numbers depend on your run). These numbers are the main takeaway of the notebook — a big drop compared to the original test accuracy (from Exercise 4) signals the model doesn't generalize well to that condition, while a similar number suggests it holds up reasonably well.
