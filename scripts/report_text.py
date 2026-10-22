"""
Text content of the SignSense report, used by scripts/build_report.py.

Content items:
    ("h1", "1. Title")                       chapter title (starts a new page)
    ("h2", "1.1 Title")                      sub-title
    ("p", "text with **bold** parts")        paragraph
    ("bullets", ["item", ...])               bulleted list
    ("figure", path, width_inches, caption[, max_height_inches])  image from the project folder
    ("screenshot", file, description, caption)
    ("table", caption, rows, column_widths)  first row = header
Edit the text here and rebuild the report.
"""

ABSTRACT = [
    "Traffic sign recognition is a key part of driver assistance systems and self-driving cars. "
    "This project, SignSense, classifies images of road traffic signs into 43 classes using deep "
    "learning on the German Traffic Sign Recognition Benchmark (GTSRB), which contains 39,209 "
    "training and 12,630 test images. We designed a compact convolutional neural network (CNN) "
    "trained from scratch on 32x32 images with data augmentation, dropout and batch normalization, "
    "and compared it with transfer learning using MobileNetV2 pre-trained on ImageNet. The custom "
    "CNN with batch normalization achieved 98.60% accuracy on the official test set using only "
    "487,243 parameters. MobileNetV2 reached 88.30% after fine-tuning its top 60 layers, and only "
    "66.75% with light fine-tuning. Error analysis shows that most mistakes involve triangular "
    "warning signs with small, similar symbols and rare classes. A Streamlit web application shows "
    "the model's prediction, confidence and top-3 predictions for unseen test images."
]

CHAPTERS = [
    # ---------------------------------------------------------------
    ("h1", "1. Introduction"),
    ("h2", "1.1 Motivation"),
    ("p", "Road traffic signs inform drivers about speed limits, dangers, priorities and mandatory "
          "directions. Missing or misreading a sign can cause accidents, especially at night, in bad "
          "weather or when the driver is tired. Advanced Driver Assistance Systems (ADAS) and "
          "self-driving cars therefore need to recognise traffic signs automatically, quickly and "
          "reliably."),
    ("p", "The task is harder than it looks. Real photos vary in lighting, blur, viewing angle and "
          "distance, and many signs look alike: speed limits differ by a single digit, and many "
          "warning signs share the same red triangle and differ only in a small black symbol in the "
          "middle."),
    ("h2", "1.2 Background"),
    ("p", "Convolutional Neural Networks (CNNs) learn visual features such as edges, shapes and "
          "symbols directly from data, and have become the standard approach for image "
          "classification. The GTSRB benchmark [1] showed that machine learning methods can match "
          "human performance on traffic sign recognition. Transfer learning reuses a network "
          "pre-trained on a large dataset such as ImageNet; MobileNetV2 [2] is a lightweight network "
          "designed for mobile devices. Our models also use batch normalization [3], dropout [4] and "
          "the Adam optimizer [5]."),
    ("p", "This project builds a small CNN from scratch, compares it with MobileNetV2 transfer "
          "learning, analyses the errors, and provides a web demo that predicts the sign in unseen "
          "test images."),

    # ---------------------------------------------------------------
    ("h1", "2. Problem Statement"),
    ("p", "Build a deep learning system that takes an image of a road traffic sign and correctly "
          "classifies it into one of 43 classes (stop, yield, speed limits, turn signs, etc.), with "
          "high accuracy, proper evaluation and an easy-to-use demo."),
    ("h2", "2.1 Objectives"),
    ("bullets", [
        "Build and train a custom CNN from scratch that reaches at least 95% test accuracy.",
        "Build a second model using transfer learning (MobileNetV2) and compare it with the custom CNN.",
        "Evaluate both models with accuracy and loss curves, a confusion matrix, per-class "
        "precision, recall and F1-score, and analysis of wrong predictions.",
        "Build a web demo that shows the predicted sign and its confidence for test images.",
        "Produce clean, documented and reproducible code.",
    ]),
    ("h2", "2.2 Dataset"),
    ("p", "We use the German Traffic Sign Recognition Benchmark (GTSRB) [1] from Kaggle [6]. It "
          "contains real photos taken from a moving car, cropped around the sign."),
    ("bullets", [
        "**Classes:** 43 sign types (speed limits, prohibitions, warnings, mandatory directions).",
        "**Size:** 39,209 training images and 12,630 test images; image sides range from "
        "25 to 266 pixels, and many are dark or blurry.",
        "**Class imbalance:** from 210 images (Speed limit 20 km/h) to 2,250 images (Speed limit "
        "50 km/h) per class, with a mean of 912 (Fig. 2.1).",
        "**Split:** 80% training (31,367 images) and 20% validation (7,842 images), stratified by "
        "class. The official test set is used only for the final evaluation.",
    ]),
    ("figure", "results/class_distribution.png", 6.2, "Fig. 2.1: Number of training images per class"),

    # ---------------------------------------------------------------
    ("h1", "3. Proposed System"),
    ("h2", "3.1 Block diagram, its description and working"),
    ("figure", "results/block_diagram.png", 6.2, "Fig. 3.1: Block diagram of SignSense"),
    ("p", "**Working:** An input image (from the dataset, or a test image chosen in the app) is read with OpenCV, "
          "converted from BGR to RGB and resized to 32x32 pixels for the custom CNN or 96x96 pixels "
          "for MobileNetV2. Pixel values are scaled (divided by 255 for the CNN; scaled to -1..1 for "
          "MobileNetV2) and labels are one-hot encoded."),
    ("p", "During training only, images are augmented with random rotation (10 degrees), zoom (15%), "
          "shift (10%) and shear. Images are never flipped, because flipping changes the meaning of a "
          "sign: “Turn left ahead” would become “Turn right ahead” (Fig. 3.2)."),
    ("figure", "results/augmentation_examples.png", 6.2,
     "Fig. 3.2: A “Turn left ahead” sign and eight augmented versions (no flips)"),
    ("p", "The image is then passed to the classifier. **Model A** is our custom CNN: two blocks of "
          "two 3x3 convolutions (32 and 64 filters) with batch normalization, max pooling and dropout "
          "(0.25), followed by Flatten, Dense(256), Dropout(0.5) and a 43-way softmax layer "
          "(487,243 parameters). **Model B** is MobileNetV2 pre-trained on ImageNet with a new head "
          "(GlobalAveragePooling, Dropout(0.3), Dense(43)); it is first trained with the base frozen "
          "and then fine-tuned by unfreezing the top 60 layers (2,313,067 parameters)."),
    ("p", "The softmax layer outputs a probability for each of the 43 classes. The class with the "
          "highest probability is the prediction and its probability is the confidence. The web app "
          "shows the sign name, the confidence, the top-3 predictions and a warning when the "
          "confidence is below 50%."),
    ("h2", "3.2 Module Description"),
    ("p", "The code is organised as a small Python package. Each module has one clear job:"),
    ("table", "Table 3.1: Project modules", [
        ["Module", "Description"],
        ["config.py", "All paths and hyperparameters (image size, batch size, epochs, learning rates)."],
        ["data_loader.py", "Loads images with OpenCV (BGR to RGB, resize) and labels; caches them as .npz."],
        ["preprocess.py", "Normalisation, one-hot labels, stratified 80/20 split, augmentation, tf.data pipeline."],
        ["models.py", "build_custom_cnn() for Model A and build_mobilenet() for Model B."],
        ["train.py", "Trains a model with EarlyStopping, ModelCheckpoint and ReduceLROnPlateau; saves history."],
        ["evaluate.py", "Test accuracy, training curves, confusion matrix, per-class report, wrong predictions."],
        ["compare.py", "Comparison table and chart of all trained models."],
        ["predict.py", "predict_image(): predicts the sign in one image with the training preprocessing."],
        ["app/app.py", "Streamlit web app: random test image, model choice, prediction, confidence, top-3 chart."],
    ], [2000, 7000]),

    # ---------------------------------------------------------------
    ("h1", "4. Implementation Result and Analysis"),
    ("p", "The project was implemented in Python with TensorFlow / Keras, OpenCV, NumPy, pandas, "
          "scikit-learn, Matplotlib, Seaborn and Streamlit. The custom CNN was trained on a laptop CPU "
          "(Python 3.11, TensorFlow 2.16) and MobileNetV2 on a Google Colab T4 GPU."),
    ("h2", "4.1 Implementation Screenshots"),
    ("screenshot", "app_home.png", "home page of the SignSense web app",
     "Fig. 4.1: SignSense web app: home page"),
    ("screenshot", "app_prediction.png", "the app showing a prediction with the top-3 chart",
     "Fig. 4.2: SignSense web app: prediction with confidence and top-3 chart"),
    ("h2", "4.2 Training and Test Results"),
    ("table", "Table 4.1: Training settings", [
        ["Setting", "Model A: Custom CNN", "Model B: MobileNetV2"],
        ["Input size", "32 x 32 x 3", "96 x 96 x 3"],
        ["Optimizer, learning rate", "Adam, 0.001", "Adam, 0.001 then 0.0001"],
        ["Epochs, batch size", "up to 20, 64", "10 + 15 (fine-tuning), 64"],
        ["Training time", "12.5 min (CPU)", "8.9 min (T4 GPU)"],
    ], [2800, 3100, 3100]),
    ("figure", "results/curves_custom_cnn_bn.png", 6.2,
     "Fig. 4.3: Training and validation accuracy and loss of Model A"),
    ("p", "Model A reached **98.60% accuracy on the official test set** (177 wrong out of 12,630), "
          "above the 95% target. Validation accuracy (99.92%) is higher than test accuracy because "
          "the training data contains “tracks” of about 30 near-identical photos of each "
          "physical sign, so a random split puts similar images in both training and validation. "
          "Training accuracy stays slightly below validation accuracy because training images are "
          "augmented and dropout is active only during training. Adding batch normalization to the "
          "baseline CNN raised test accuracy from 98.05% to 98.60% for only 768 extra parameters."),
    ("h2", "4.3 Comparison of Model A and Model B"),
    ("table", "Table 4.2: Comparison on the official test set (12,630 images)", [
        ["Model", "Parameters", "Test accuracy", "Macro-F1", "File size"],
        ["Custom CNN (baseline)", "486,475", "98.05%", "0.963", "5.9 MB"],
        ["Custom CNN + BatchNorm (Model A)", "487,243", "98.60%", "0.973", "5.9 MB"],
        ["MobileNetV2, light fine-tuning", "2,313,067", "66.75%", "-", "-"],
        ["MobileNetV2, fine-tuned (Model B)", "2,313,067", "88.30%", "0.842", "26.0 MB"],
    ], [3200, 1500, 1500, 1300, 1500]),
    ("figure", "results/curves_mobilenetv2.png", 6.2,
     "Fig. 4.4: MobileNetV2 training: fine-tuning starts at epoch 11"),
    ("p", "The small custom CNN is more accurate than MobileNetV2 while being about 5 times smaller. "
          "MobileNetV2 was pre-trained on large, detailed ImageNet photos, while GTSRB images are tiny "
          "and blurry. Its GlobalAveragePooling layer also discards the position of features, so "
          "left and right signs are confused (“Go straight or right” was predicted as "
          "“Go straight or left” 33 times). Fine-tuning was essential: with the base "
          "frozen, validation accuracy stalled at about 77%, and it jumped to 97% once the top 60 "
          "layers were unfrozen (Fig. 4.4)."),
    ("h2", "4.4 Error Analysis"),
    ("table", "Table 4.3: Five classes with the lowest F1-score (Model A)", [
        ["Class", "F1", "Recall", "Main confusion"],
        ["27 Pedestrians", "0.66", "50%", "predicted as Children crossing (25 of 60)"],
        ["22 Bumpy road", "0.86", "75%", "predicted as Road work (28 of 120)"],
        ["40 Roundabout mandatory", "0.89", "99%", "Priority road predicted as Roundabout (19)"],
        ["28 Children crossing", "0.92", "100%", "receives the Pedestrians errors (25)"],
        ["21 Double curve", "0.93", "100%", "General caution predicted as Double curve (13)"],
    ], [2600, 900, 1000, 4500]),
    ("figure", "results/misclassified_custom_cnn_bn.png", 6.2,
     "Fig. 4.5: Examples of misclassified test images (Model A)"),
    ("p", "Most errors are between triangular warning signs with the same red border, whose only "
          "difference is a small black symbol that is just a few pixels wide at 32x32. Several of "
          "these classes are rare (Pedestrians has only 240 training images), and many wrong images "
          "are very dark, blurred or overexposed."),

    # ---------------------------------------------------------------
    ("h1", "5. Code"),
    ("p", "The most important parts of the code are shown below. The full code is in the project "
          "repository and in the Colab notebook."),
    ("figure", "results/code/code_custom_cnn.png", 6.2, "Fig. 5.1: Model A, custom CNN (models.py)", 3.9),
    ("figure", "results/code/code_mobilenet.png", 6.2,
     "Fig. 5.2: Model B, MobileNetV2 transfer learning (models.py)", 3.9),
    ("figure", "results/code/code_training.png", 6.2, "Fig. 5.3: Training with callbacks (train.py)", 3.9),
    ("figure", "results/code/code_predict.png", 6.2, "Fig. 5.4: Predicting a single image (predict.py)", 3.9),

    # ---------------------------------------------------------------
    ("h1", "6. Conclusion"),
    ("p", "We built SignSense, a traffic sign recognition system for the 43 GTSRB classes. A compact "
          "custom CNN with data augmentation (without flips), dropout and batch normalization reached "
          "**98.60% test accuracy** with fewer than 0.5 million parameters, above the 95% target. "
          "MobileNetV2 transfer learning reached 88.30% after fine-tuning, showing that pre-trained "
          "ImageNet features do not transfer well to tiny, low-resolution sign images and that "
          "fine-tuning is essential. The remaining errors come mainly from similar triangular warning "
          "signs, rare classes and poor image quality. A Streamlit web app demonstrates the model on "
          "unseen test images."),
    ("h2", "6.1 Limitations and Future Scope"),
    ("bullets", [
        "The models only know the 43 GTSRB classes: a sign from outside these classes is still "
        "forced into the closest known class, so the demo uses test images only.",
        "The models classify already cropped signs; real-time detection in full camera frames could "
        "be added with an object detector such as YOLO.",
        "Video input from a dashcam, combining predictions over several frames.",
        "Training on more datasets, such as Indian traffic signs, to handle other countries.",
        "Deployment on mobile and edge devices with TensorFlow Lite; our model is only about 6 MB.",
        "Higher accuracy through cropping to the sign's bounding box, class weights for rare "
        "classes and higher input resolution.",
    ]),
]

REFERENCES = [
    "[1] J. Stallkamp, M. Schlipsing, J. Salmen and C. Igel, “Man vs. computer: Benchmarking "
    "machine learning algorithms for traffic sign recognition,” Neural Networks, vol. 32, "
    "pp. 323-332, 2012.",
    "[2] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov and L.-C. Chen, “MobileNetV2: Inverted "
    "residuals and linear bottlenecks,” in Proc. IEEE/CVF Conference on Computer Vision and "
    "Pattern Recognition (CVPR), 2018, pp. 4510-4520.",
    "[3] S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by "
    "reducing internal covariate shift,” in Proc. International Conference on Machine Learning "
    "(ICML), 2015, pp. 448-456.",
    "[4] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever and R. Salakhutdinov, “Dropout: "
    "A simple way to prevent neural networks from overfitting,” Journal of Machine Learning "
    "Research, vol. 15, pp. 1929-1958, 2014.",
    "[5] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in Proc. "
    "International Conference on Learning Representations (ICLR), 2015.",
    "[6] Kaggle, ‘GTSRB - German Traffic Sign Recognition Benchmark’. [Online]. Available: "
    "https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign. "
    "[Accessed: 19- Oct- 2026].",
]


def _toc_entries():
    """(number, title, level) for every heading, plus References, in document order."""
    entries = []
    for item in CHAPTERS:
        if item[0] in ("h1", "h2"):
            number, title = item[1].split(" ", 1)
            entries.append((number.rstrip("."), title, 1 if item[0] == "h1" else 2))
    entries.append(("", "References", 1))
    return entries


TOC_ENTRIES = _toc_entries()
