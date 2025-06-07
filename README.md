
# Skin Cancer Classification with ResNet50 on HAM10000 Dataset

This project demonstrates a deep learning pipeline for skin lesion classification using the HAM10000 dataset. The model leverages transfer learning with a ResNet50 backbone and is trained to classify seven different types of skin lesions.

---

## Project Overview

Skin cancer is one of the most common cancers worldwide. Early detection significantly improves treatment success rates. This project uses convolutional neural networks (CNNs) to classify skin lesion images into 7 classes:

* Actinic keratoses (akiec)
* Basal cell carcinoma (bcc)
* Benign keratosis-like lesions (bkl)
* Dermatofibroma (df)
* Melanoma (mel)
* Melanocytic nevi (nv)
* Vascular lesions (vasc)

The ResNet50 model pre-trained on ImageNet is fine-tuned on the HAM10000 dataset for this task.

---

## Dataset

* **HAM10000** (Human Against Machine with 10000 training images) skin lesion dataset.
* Contains 10,015 dermatoscopic images.
* Images are categorized into 7 diagnostic classes.
* Dataset source: [Kaggle - Skin Cancer MNIST: HAM10000](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)

---

## Installation

You need Python 3.x and the following Python packages:

```bash
pip install tensorflow keras pandas numpy matplotlib seaborn scikit-learn gradio
```

---

## Usage

### 1. Prepare Images

Copy all images into a single directory:

```bash
mkdir -p /kaggle/working/all_images
cp ../input/skin-cancer-mnist-ham10000/HAM10000_images_part_1/* /kaggle/working/all_images/
cp ../input/skin-cancer-mnist-ham10000/HAM10000_images_part_2/* /kaggle/working/all_images/
```

### 2. Load Metadata and Prepare Dataset

* Load metadata CSV containing image filenames and labels.
* Split data into train, validation, and test sets.

### 3. Build and Train Model

* Use a ResNet50 backbone (without top layers).
* Add custom layers for classification.
* Compile and train the model with augmentation.

### 4. Evaluate Model

* Generate classification report and confusion matrix.
* Visualize sample predictions.

### 5. Launch Gradio Demo

Run the Gradio interface to perform inference on uploaded images interactively.

---

## Model Architecture

* Base: ResNet50 (ImageNet weights, excluding top layers)
* Added:

  * Conv2D layer (64 filters, 3x3, ReLU)
  * MaxPooling2D
  * Dropout (0.4)
  * Flatten
  * Dense (128 units, ReLU)
  * Dropout (0.4)
  * Dense output layer (7 units, softmax)

---

## Training

* Adam optimizer with learning rate = 0.0001
* Loss: categorical crossentropy
* Metrics: accuracy
* Callbacks: ReduceLROnPlateau, EarlyStopping
* Data augmentation applied to training images

---

## Evaluation

* Achieved **\~81.7% accuracy** on test set.
* Classification report and confusion matrix generated.
* Sample predictions visualized with true vs predicted labels.

---

## Demo

A Gradio web app is included to interactively upload skin lesion images and get classification results with confidence scores.


![Benign keratosis-like lesions (bkl)](https://github.com/tejwani-rahul/skin_cancer_classifier/blob/main/Demo%20Images/BKL%20Example.png)
![Melanoma (mel)](https://github.com/tejwani-rahul/skin_cancer_classifier/blob/main/Demo%20Images/Mel%20Example.png)
![Melanocytic nevi (nv)](https://github.com/tejwani-rahul/skin_cancer_classifier/blob/main/Demo%20Images/nv%20Example.png)



---

## Results

| Metric            | Value  |
| ----------------- | ------ |
| Test Accuracy     | 81.7%  |
| Weighted F1-Score | \~0.82 |

Confusion matrix and classification reports are included in the notebook outputs.

