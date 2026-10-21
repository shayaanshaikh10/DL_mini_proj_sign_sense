# Model Comparison: Custom CNN vs MobileNetV2

All models were tested on the same official GTSRB test set (12,630 images, 43 classes),
which was never used for training or model selection.

## Comparison table

| Model | Input | Params | Train time | Hardware | Best val acc. | **Test acc.** | Macro-F1 | File size |
|---|---|---|---|---|---|---|---|---|
| Custom CNN (baseline) | 32x32 | 486,475 | 14.5 min | laptop CPU | 99.92% | 98.05% | 0.963 | 5.9 MB |
| **Custom CNN + BatchNorm (Model A)** | 32x32 | 487,243 | 12.5 min | laptop CPU | 99.92% | **98.60%** | **0.973** | 5.9 MB |
| MobileNetV2, light fine-tuning (run 1) | 96x96 | 2,313,067 | 5.6 min | Colab T4 GPU | 77.93% | 66.75% | n/a | n/a |
| MobileNetV2, fine-tuned (Model B) | 96x96 | 2,313,067 | 8.9 min | Colab T4 GPU | 97.27% | 88.30% | 0.842 | 26.0 MB |

Notes:
- Training times are not directly comparable: the CNNs trained on a laptop CPU, MobileNetV2 on a GPU.
- Run 1 fine-tuned the top 20 layers (lr 1e-5, 5 epochs); run 2 fine-tuned the top 60 layers
  (lr 1e-4, 15 epochs). Run 1's model was overwritten, so its numbers come from the training log.
- File sizes include the optimizer state saved by Keras.

## Which model is better?

**Model A (Custom CNN + BatchNorm) is the best model**: 98.60% test accuracy, against 88.30%
for the best MobileNetV2, with about 5 times fewer parameters and a model file about 4 times smaller.
It is also more consistent across classes: only 3 of 43 classes have F1 below 0.90, compared with
25 classes for MobileNetV2. MobileNetV2 beats Model A on just 3 classes (Priority road, No vehicles,
and Pedestrians by a negligible 0.001).

## Why the small CNN wins

1. **Domain mismatch.** MobileNetV2 was pre-trained on ImageNet: large, detailed photos of everyday
   objects. GTSRB images are tiny (often 30-50 pixels), blurry and sometimes very dark. Upscaling
   them to 96x96 adds no detail, so many pre-trained features are not useful here.
2. **Position matters for traffic signs.** MobileNetV2's head uses GlobalAveragePooling, which
   averages each feature map over the whole image and loses *where* a feature is. "Keep left" and
   "Keep right" contain the same parts in different places. Run 1 confused them 64 times; even after
   fine-tuning, "Go straight or left" is predicted as "Go straight or right" 33 times. The custom
   CNN uses Flatten, which keeps spatial layout.
3. **Fine-tuning is essential.** With the base frozen, validation accuracy stalled around 76-77%.
   As soon as 60 layers were unfrozen, it jumped to about 97% (see `curves_mobilenetv2.png`).
   Light fine-tuning (run 1) was not enough: 66.75% test accuracy.
4. **Overfitting.** MobileNetV2 has a 9-point gap between validation (97.3%) and test (88.3%),
   compared with 1.3 points for Model A. A larger model memorises the near-duplicate images in
   GTSRB's training "tracks" more easily.
5. **BatchNorm helps the CNN.** Adding BatchNormalization after each convolution cost only 768
   parameters but cut test errors from 246 to 177 (98.05% to 98.60%).

## What both models still get wrong

Both models struggle most with triangular warning signs whose only difference is a small black
symbol in the middle (Pedestrians vs Children crossing, Beware of ice/snow vs Bicycles crossing
or Road work, Double curve vs Dangerous curve), and with rare classes that have only about 200-250
training images. Similar speed limits (20 vs 50, 60 vs 80 km/h) are also confused.

## Conclusion

For a small, low-resolution dataset like GTSRB, a compact CNN designed for the task beats a large
pre-trained network: it is more accurate, smaller and faster. Transfer learning is not
automatically better; it needs heavy fine-tuning here and still does not catch up.
Model A is used in the demo app.
