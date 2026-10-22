# SignSense: Report and PPT Content

Ready-to-paste text for the project report and slides. All numbers come from our own runs
(files in `results/`). Figure file names are given in *[Figure: ...]* notes.

> **Team, please check before submitting:** every reference in Section 4 must be verified
> against the original paper (authors, year, venue and the numbers quoted). We have marked
> them **[verify]**. Do not add references that you have not checked yourselves.

---

## 1. Abstract

Traffic sign recognition is a key part of driver assistance systems and self-driving cars.
This project, SignSense, classifies images of road signs into 43 classes using deep learning
on the German Traffic Sign Recognition Benchmark (GTSRB), which contains 39,209 training and
12,630 test images. We designed a compact convolutional neural network (CNN) trained from
scratch on 32x32 images with data augmentation, dropout and batch normalization, and compared
it with transfer learning using MobileNetV2 pre-trained on ImageNet. The custom CNN with batch
normalization achieved 98.60% accuracy on the official test set using only 487,000 parameters.
MobileNetV2 reached 88.30% after fine-tuning its top 60 layers, and only 66.75% with light
fine-tuning. Error analysis shows that most mistakes involve triangular warning signs with
small, similar symbols and rare classes. A Streamlit web application shows the predicted
class, confidence and top-3 predictions for unseen test images.

*(about 150 words)*

---

## 2. Introduction and Motivation

Road traffic signs tell drivers about speed limits, dangers, priorities and mandatory
directions. Missing or misreading a sign can cause accidents, especially when drivers are
tired or distracted, at night, or in bad weather. Advanced Driver Assistance Systems (ADAS)
and self-driving cars therefore need to recognise traffic signs automatically, quickly and
reliably.

Recognising signs is harder than it looks. Real photos vary in lighting (dark, overexposed),
blur from motion, viewing angle, distance and partial occlusion. Many signs also look alike:
several speed limits differ by only one digit, and many warning signs share the same red
triangle and differ only in a small black symbol.

Deep learning, and convolutional neural networks (CNNs) in particular, have become the
standard approach for image classification because they learn useful visual features
directly from data instead of relying on hand-crafted features. This project studies how
well a small CNN trained from scratch performs on this task, and whether transfer learning
from a large pre-trained network (MobileNetV2) helps.

Applications: driver assistance (sign alerts on the dashboard), autonomous vehicles, road
infrastructure surveys (checking that signs are present and visible), and driver training.

---

## 3. Problem Statement and Objectives

**Problem statement:** Build a deep learning system that takes an image of a road traffic sign
and correctly classifies it into one of 43 classes, with high accuracy, proper evaluation and
an easy-to-use demo.

**Objectives:**
1. Build and train a custom CNN from scratch that reaches at least 95% test accuracy.
2. Build a second model using transfer learning (MobileNetV2) and compare it with the custom CNN.
3. Evaluate properly: accuracy and loss curves, confusion matrix, per-class precision, recall
   and F1, and analysis of wrong predictions.
4. Build a web demo that shows the predicted sign and its confidence for test images.
5. Produce clean, documented and reproducible code.

**Outcome:** all objectives were met. The custom CNN reached 98.05% (baseline) and 98.60%
(with batch normalization), above both the 95% target and the 97% stretch goal.

---

## 4. Literature Survey

> All entries below are **[verify]**: check each against the original publication before use.

**4.1 Stallkamp, Schlipsing, Salmen and Igel (2012), "Man vs. computer: Benchmarking machine
learning algorithms for traffic sign recognition", *Neural Networks*, 32, 323-332. [verify]**
This paper introduced the GTSRB dataset used in our project and summarised the 2011 IJCNN
competition. It compared machine learning methods with human performance on the same images;
the best methods matched or exceeded the human recognition rate (reported as about 98.8%) [verify].
It established GTSRB as the standard benchmark for traffic sign classification.

**4.2 Cireşan, Meier, Masci and Schmidhuber (2012), "Multi-column deep neural network for
traffic sign classification", *Neural Networks*, 32, 333-338. [verify]**
The winning approach of the GTSRB competition. It combined several deep CNNs ("columns"),
each trained on differently preprocessed images, and averaged their outputs. It reported about
99.46% test accuracy, better than human performance [verify]. This showed the strength of deep
CNNs for this task, at the cost of a large ensemble and long GPU training.

**4.3 Sermanet and LeCun (2011), "Traffic sign recognition with multi-scale convolutional
networks", *International Joint Conference on Neural Networks (IJCNN)*. [verify]**
Proposed a CNN in which features from both early and late layers are fed to the classifier
(multi-scale features), so the network uses both fine details and global shape. It achieved
around 99% accuracy on GTSRB [verify], showing that a single well-designed CNN can perform
very well.

**4.4 Sandler, Howard, Zhu, Zhmoginov and Chen (2018), "MobileNetV2: Inverted Residuals and
Linear Bottlenecks", *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. [verify]**
Introduced MobileNetV2, a lightweight CNN built from inverted residual blocks with depthwise
separable convolutions, designed for mobile and embedded devices. We use its ImageNet-pretrained
weights as Model B for transfer learning.

**4.5 Ioffe and Szegedy (2015), "Batch Normalization: Accelerating Deep Network Training by
Reducing Internal Covariate Shift", *International Conference on Machine Learning (ICML)*. [verify]**
Proposed normalising layer activations within each mini-batch, which stabilises and speeds up
training and acts as a mild regulariser. We added batch normalization to our CNN and observed
a test accuracy gain from 98.05% to 98.60%.

**Summary of gap:** published state-of-the-art results use large ensembles or complex
architectures. Our project shows that a simple, single CNN with standard techniques
(augmentation, dropout, batch normalization) reaches 98.6% with modest compute, and examines
why generic transfer learning does not automatically help on this dataset.

---

## 5. Dataset Description

- **Name:** GTSRB, German Traffic Sign Recognition Benchmark (Kaggle version).
- **Classes:** 43 sign types: speed limits, prohibitions, warnings, mandatory directions and
  others (full list in `data/signnames.csv`).
- **Size:** 39,209 training images and 12,630 test images.
- **Image properties:** real photos from a car-mounted camera, cropped around the sign; sizes
  sides range from 25 to 266 pixels; many images are dark, blurry or low contrast.
- **Class imbalance:** between 210 images (e.g. Speed limit 20 km/h) and 2,250 images
  (Speed limit 50 km/h) per class; mean 912. *[Figure: results/class_distribution.png]*
- **Tracks:** the training set contains "tracks" of about 30 photos of the same physical sign
  taken as the car approaches. Images within a track are nearly identical.
- **Our split:** 80% training (31,367 images) and 20% validation (7,842 images), stratified by
  class with `random_state=42`. The official test set (12,630 images, labels in `Test.csv`) was
  used only for final evaluation.

*[Figure: results/sample_images.png: random training samples with their names]*

---

## 6. Methodology

### 6.1 Pipeline overview

Image -> load with OpenCV -> convert BGR to RGB -> resize (32x32 for the CNN, 96x96 for
MobileNetV2) -> scale pixels -> model -> softmax probabilities over 43 classes -> predicted class.

### 6.2 Preprocessing

- Pixel values converted to float32 and divided by 255 (range 0-1) for the custom CNN.
- For MobileNetV2, MobileNetV2's own `preprocess_input` scales pixels to the range -1 to 1.
- Labels one-hot encoded into vectors of length 43.

### 6.3 Data augmentation

Applied to training images only, with new random transformations every epoch:
rotation up to 10 degrees, zoom up to 15%, horizontal and vertical shift up to 10%, shear up to 0.1.
**No horizontal or vertical flips**, because flipping changes the meaning of a sign
("Turn left ahead" would become "Turn right ahead"). *[Figure: results/augmentation_examples.png]*

### 6.4 Model A: Custom CNN (architecture in text)

```
Input 32 x 32 x 3
Block 1: Conv2D 32 filters 3x3, ReLU -> BatchNorm -> Conv2D 32 filters 3x3, ReLU -> BatchNorm
         -> MaxPooling 2x2 -> Dropout 0.25                         (output 14 x 14 x 32)
Block 2: Conv2D 64 filters 3x3, ReLU -> BatchNorm -> Conv2D 64 filters 3x3, ReLU -> BatchNorm
         -> MaxPooling 2x2 -> Dropout 0.25                         (output 5 x 5 x 64)
Classifier: Flatten (1,600) -> Dense 256, ReLU -> Dropout 0.5 -> Dense 43, Softmax
Total parameters: 487,243 (baseline without BatchNorm: 486,475)
```

The convolution blocks learn features from simple (edges, colours) to complex (sign shapes and
symbols). Max pooling reduces the image size and adds tolerance to small shifts. Dropout reduces
overfitting. Flatten keeps the spatial layout of the features, which matters for direction signs.

### 6.5 Model B: MobileNetV2 transfer learning (architecture in text)

```
Input 96 x 96 x 3
MobileNetV2 base, pre-trained on ImageNet, top removed (output 3 x 3 x 1280)
GlobalAveragePooling2D (1,280) -> Dropout 0.3 -> Dense 43, Softmax
Total parameters: 2,313,067
```

Training in two stages:
1. **Feature extraction:** base frozen, only the new head is trained (10 epochs, learning rate 0.001).
2. **Fine-tuning:** the top 60 layers of the base are unfrozen and trained with a small learning
   rate (0.0001) for 15 epochs. Batch normalization layers in the base stay in inference mode.

Images are fed through a `tf.data` pipeline that converts and augments them batch by batch,
so the 96x96 dataset does not need to be stored in memory as floating point numbers.

### 6.6 Training setup

- Loss: categorical cross-entropy. Metric: accuracy. Optimizer: Adam.
- Callbacks: EarlyStopping (patience 5 on validation loss, restoring the best weights),
  ModelCheckpoint (saves the model with the best validation accuracy) and ReduceLROnPlateau
  (halves the learning rate after 3 epochs without improvement).

---

## 7. Implementation Details

**Tools:** Python 3.11, TensorFlow 2.16 / Keras 3, NumPy, pandas, OpenCV, scikit-learn,
Pillow, Matplotlib, Seaborn, Streamlit. Custom CNN trained on a laptop CPU; MobileNetV2
trained on a Google Colab T4 GPU.

| Hyperparameter | Custom CNN | MobileNetV2 |
|---|---|---|
| Input size | 32 x 32 x 3 | 96 x 96 x 3 |
| Batch size | 64 | 64 |
| Optimizer / learning rate | Adam, 0.001 | Adam, 0.001 (head), 0.0001 (fine-tuning) |
| Epochs | up to 20 | 10 (head) + 15 (fine-tuning) |
| Dropout | 0.25 (conv), 0.5 (dense) | 0.3 |
| Early stopping patience | 5 | 5 |
| Random seed | 42 | 42 |
| Training time | about 12-15 min (CPU) | 8.9 min (T4 GPU) |

**Code structure:** `src/config.py` (all settings), `data_loader.py`, `preprocess.py`,
`models.py`, `train.py`, `evaluate.py`, `compare.py`, `predict.py`, and the Streamlit app in
`app/app.py`. A Colab notebook (`notebooks/signsense_colab.ipynb`) runs the whole pipeline.

---

## 8. Results and Discussion

### 8.1 Overall results (official test set, 12,630 images)

| Model | Test accuracy | Wrong predictions | Macro-F1 |
|---|---|---|---|
| Custom CNN (baseline) | 98.05% | 246 | 0.963 |
| **Custom CNN + BatchNorm (Model A)** | **98.60%** | **177** | **0.973** |
| MobileNetV2, fine-tuned (Model B) | 88.30% | 1,478 | 0.842 |

### 8.2 Training curves

*[Figure: results/curves_custom_cnn_bn.png]* Training and validation accuracy rise quickly:
validation accuracy passes 98% by epoch 3-4 and reaches 99.92%. Training accuracy (99.3% at the
end) stays slightly below validation accuracy because training images are augmented and dropout
is active during training, which makes training harder than validation. There is no sign of
overfitting: validation loss keeps decreasing.

**Validation vs test:** validation accuracy (99.92%) is higher than test accuracy (98.60%)
because of GTSRB's tracks. A random split puts near-identical photos of the same physical sign
in both training and validation, so validation is easier. The test set contains different
physical signs, so it is the honest measure of performance.

### 8.3 Confusion matrix

*[Figure: results/confusion_matrix_custom_cnn_bn.png and results/confusion_errors_custom_cnn_bn.png]*
The row-normalised matrix is almost perfectly diagonal. The "errors only" version (diagonal
removed) shows that mistakes are concentrated in a few pairs of similar signs.

### 8.4 Error analysis: 5 worst classes for Model A (lowest F1)

| Class | F1 | Recall | Main confusion |
|---|---|---|---|
| 27 Pedestrians | 0.66 | 50% | predicted as Children crossing (25 of 60) |
| 22 Bumpy road | 0.86 | 75% | predicted as Road work (28 of 120) |
| 40 Roundabout mandatory | 0.89 | 99% | Priority road images predicted as Roundabout (19) |
| 28 Children crossing | 0.92 | 100% | receives the Pedestrians errors (25) |
| 21 Double curve | 0.93 | 100% | General caution images predicted as Double curve (13) |

**Likely reasons:**
- **Similar-looking signs:** most errors are between triangular warning signs with the same red
  border and white background, which differ only in a small black symbol. At 32x32 pixels that
  symbol is only a few pixels wide.
- **Rare classes:** Pedestrians has only 240 training images, compared with over 2,000 for the
  most common classes.
- **Image quality:** several misclassified images are very dark, blurred or overexposed.
  *[Figure: results/misclassified_custom_cnn_bn.png]*

### 8.5 Effect of batch normalization

Adding batch normalization after each convolution (768 extra parameters) reduced test errors
from 246 to 177 and raised macro-F1 from 0.963 to 0.973, with the same training settings. It
also improved validation accuracy faster in the first epochs. We therefore use this version as Model A.

---

## 9. Comparison: Custom CNN vs MobileNetV2

| | Model A: Custom CNN + BN | Model B: MobileNetV2 |
|---|---|---|
| Input size | 32 x 32 | 96 x 96 |
| Parameters | 487,243 | 2,313,067 (4.7x more) |
| Model file | 5.9 MB | 26.0 MB |
| Best validation accuracy | 99.92% | 97.27% |
| **Test accuracy** | **98.60%** | **88.30%** |
| Classes with F1 below 0.90 | 3 | 25 |

*[Figure: results/comparison_chart.png, results/curves_mobilenetv2.png]*

**Discussion:**
1. **The custom CNN is better** in accuracy, size and consistency across classes. MobileNetV2
   is better on only 3 of 43 classes.
2. **Domain mismatch:** MobileNetV2 learned its features on ImageNet, which contains large,
   detailed photos of everyday objects. GTSRB images are small, blurry and upscaled to 96x96, so
   many pre-trained features are not useful.
3. **Spatial information:** GlobalAveragePooling averages each feature map over the whole image,
   which discards where features are located. "Keep left" and "Keep right" contain the same parts
   in different positions. With light fine-tuning, Keep left was predicted as Keep right 64 times;
   even after fine-tuning, "Go straight or right" was predicted as "Go straight or left" 33 times.
   The custom CNN uses Flatten and keeps the spatial layout.
4. **Fine-tuning is essential:** with the base frozen, validation accuracy levelled off at about
   76-77%. When the top 60 layers were unfrozen, it jumped to 89% in one epoch and then to 97%.
   A light fine-tuning (top 20 layers, learning rate 0.00001, 5 epochs) reached only 66.75% on
   the test set.
5. **Overfitting:** MobileNetV2 shows a 9-point gap between validation (97.3%) and test (88.3%),
   compared with 1.3 points for the custom CNN. The larger model memorises the near-duplicate
   track images more easily.

**Conclusion of the comparison:** transfer learning is not automatically better. For a small,
low-resolution and specialised dataset, a compact CNN designed for the task outperforms a large
general-purpose pre-trained network.

---

## 10. Conclusion

We built SignSense, a traffic sign recognition system for the 43 GTSRB classes. A custom CNN with
two convolution blocks, data augmentation (without flips), dropout and batch normalization reached
**98.60% test accuracy** with fewer than 0.5 million parameters, above our 95% target and 97%
stretch goal. Batch normalization improved the baseline from 98.05% to 98.60%. MobileNetV2 transfer
learning reached 88.30% after fine-tuning, showing that pre-trained ImageNet features do not
transfer well to tiny, low-resolution sign images and that fine-tuning is essential. Error analysis
showed that the remaining mistakes come mainly from triangular warning signs with similar symbols,
rare classes and poor image quality. A Streamlit web app demonstrates the model on unseen test
images with confidence scores and top-3 predictions.

---

## 11. Limitations and Future Scope

**Limitations**
- The models only know the **43 GTSRB classes**. A sign from outside these classes (for
  example a sign used only in other countries) is still forced into the closest known class,
  often with high confidence. For this reason the demo app uses test images only.
- The models classify **already cropped** sign images. They cannot find a sign inside a full
  street photo.
- Trained only on German signs; signs from other countries (for example India) look different.
- Low resolution (32x32) loses fine details of warning-sign symbols.
- Rare classes and visually similar warning signs remain the main source of errors.
- The validation split is affected by near-duplicate images (tracks), so validation accuracy
  overestimates real performance.

**Future scope**
- **Real-time detection:** combine the classifier with an object detector such as YOLO to locate
  signs in full camera frames, then classify them.
- **Video input:** process dashcam video and combine predictions over several frames for more
  reliable results.
- **More datasets:** train and test on other datasets, such as Indian traffic sign datasets, to
  handle signs from different countries.
- **Deployment on mobile or edge devices:** convert the model to TensorFlow Lite and run it on a
  phone or a Raspberry Pi. Our model is small (about 6 MB), which makes this practical.
- **Better accuracy:** crop to the sign's bounding box (provided in the dataset), use class
  weights or oversampling for rare classes, use a higher input resolution, or ensemble several CNNs.
- **Track-based validation split** to get a more realistic validation score.

---

## 12. Viva Questions and Answers

**1. What is a CNN?**
A Convolutional Neural Network is a neural network designed for images. Its convolution layers
slide small filters over the image to detect local patterns such as edges and shapes, and deeper
layers combine them into more complex features. It shares the same filter weights across the
image, so it needs far fewer parameters than a fully connected network.

**2. What does a convolution layer and a pooling layer do?**
A convolution layer applies learned filters (here 3x3) to produce feature maps showing where a
pattern appears. Max pooling (2x2) keeps the strongest value in each small window, halving the
width and height. This reduces computation and makes the network tolerant to small shifts.

**3. Why do we use ReLU?**
ReLU outputs max(0, x). It adds non-linearity so the network can learn complex functions, is very
cheap to compute and avoids the vanishing gradient problem of sigmoid/tanh in deep networks.

**4. What is dropout and why did you use it?**
Dropout randomly switches off a fraction of neurons during training (25% after conv blocks, 50%
before the output). The network cannot rely on any single neuron, which reduces overfitting.
Dropout is turned off during prediction.

**5. What is batch normalization and did it help?**
It normalises the outputs of a layer within each mini-batch to a stable mean and variance, then
scales and shifts them with learned parameters. It makes training more stable and faster and
acts as a mild regulariser. In our project it improved test accuracy from 98.05% to 98.60%.

**6. What is softmax?**
The last layer's activation. It turns the 43 output scores into probabilities that are all
positive and add up to 1. The class with the highest probability is the prediction, and that
probability is the confidence shown in our app.

**7. Why categorical cross-entropy loss?**
It is the standard loss for multi-class classification with one-hot labels. It measures how
far the predicted probability distribution is from the true one, and it heavily penalises
confident wrong predictions.

**8. Why the Adam optimizer?**
Adam adapts the learning rate for each weight using running averages of the gradients and their
squares. It converges quickly with little tuning; we used a learning rate of 0.001 and halved it
when validation loss stopped improving.

**9. What is data augmentation and why no flipping?**
Augmentation creates new training images by small random changes (rotation, zoom, shift, shear),
so the model learns to handle different angles and positions and overfits less. We did not flip
images because a flipped sign can mean something different: "Turn left ahead" would become
"Turn right ahead" while keeping the wrong label.

**10. What is overfitting and how did you prevent it?**
Overfitting is when a model memorises the training data and performs worse on new data. We used
data augmentation, dropout, batch normalization, early stopping and a separate validation set.
We checked for it by comparing training and validation curves and by testing on a separate test set.

**11. What is transfer learning?**
Reusing a model trained on a large dataset (MobileNetV2 on ImageNet, 1.2 million images) as a
starting point for a new task. We removed its original classifier, added a new head for 43
classes and trained that head, keeping the pre-trained features.

**12. What is fine-tuning and why use a small learning rate?**
Fine-tuning unfreezes some of the pre-trained layers and trains them on the new data, so the
features adapt to the new task. A small learning rate (0.0001) changes the pre-trained weights
gently instead of destroying them. In our project fine-tuning raised MobileNetV2 from about 77%
to 97% validation accuracy.

**13. Why did your small CNN beat MobileNetV2?**
GTSRB images are tiny and blurry, unlike ImageNet photos, so pre-trained features transfer
poorly. MobileNetV2's GlobalAveragePooling discards position information, which confuses left and
right signs. The larger model also overfits more on near-duplicate training images.

**14. Why is validation accuracy higher than test accuracy?**
The training set contains tracks of about 30 near-identical photos of the same sign. A random
80/20 split places similar images in both training and validation, making validation too easy.
The official test set contains different physical signs, so test accuracy (98.60%) is the honest number.

**15. Explain precision, recall, F1-score and the confusion matrix.**
For a class, precision = of all images predicted as that class, how many were correct; recall =
of all images that truly belong to that class, how many were found; F1 = the harmonic mean of
precision and recall. The confusion matrix is a 43x43 table whose rows are true classes and
columns are predicted classes; correct predictions lie on the diagonal and off-diagonal cells show
which classes are confused (for example Pedestrians predicted as Children crossing).

**Bonus: How did you handle class imbalance?**
We used a stratified train/validation split so every class keeps its proportion, and reported
per-class F1 and macro-F1, not only accuracy. The rarest classes are among the worst performers;
class weights or oversampling are listed as future improvements.

**Bonus: What does EarlyStopping do?**
It stops training when validation loss has not improved for 5 epochs and restores the best
weights, saving time and preventing overfitting.
