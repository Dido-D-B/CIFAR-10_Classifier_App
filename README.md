# CIFAR-10 Image Classifier App

<img width="958" height="536" alt="Screenshot 2025-09-25 at 23 01 19" src="https://github.com/user-attachments/assets/51782863-5787-4f05-8664-d891feb52114" />

---

This project demonstrates how **transfer learning with ResNet50** can classify images from the CIFAR-10 dataset into 10 categories. The final model is deployed as a Streamlit web app for interactive testing.

## Project Overview

* **Objective**: Build and deploy an image classifier for CIFAR-10 (10 classes, 60k images).
* **Approach**:
  * Used ResNet50 pre-trained on ImageNet.
  * Fine-tuned on 10k training samples with data augmentation.
  * Applied layer-wise unfreezing and learning rate scheduling.
  * Deployed model in a Streamlit app with visualization tools.

## Dataset

* **CIFAR-10**: 60,000 32×32 RGB images, 10 classes (e.g., airplane, cat, dog, ship).
* **Training set**: 50,000 images
* **Test set**: 10,000 images

## Model Architecture

* **Base**: ResNet50 (pre-trained on ImageNet)
* **Custom head**: Dense layers + BatchNorm + Dropout + Softmax
* **Techniques**:
  * Label smoothing
  * Early stopping & ReduceLROnPlateau
  * Cosine decay learning rate schedule

## Results

* Transfer learning worked well even with limited data (10k samples) ￼
* Achieved strong generalization through data augmentation & careful fine-tuning
* Visualization of metrics (accuracy/loss, confusion matrix) provided key insights

## Key Metrics

* **Accuracy**: (insert your final % here if available)
* **Challenges**:
  * Small input size (32×32 vs 224×224 expected by ResNet50)
  * Confusing classes (cat/dog/bird overlap)
  * Careful tuning of learning rates required

## Demo App

The model is deployed with Streamlit 👉 Check out the [Live Demo](https://image-aiclassifier.streamlit.app/)

<img width="1418" height="1168" alt="image" src="https://github.com/user-attachments/assets/cdf99009-e7ae-45c1-825e-87ac474b5c79" />


## Run locally

1. Clone repo

```
git clone https://github.com/Dido-D-B/CIFAR-10_Classifier_App.git
cd CIFAR-10_Classifier_App
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run Streamlit app

```
streamlit run app/Image_Classifier_Streamlit.py
```

## Project Structure

```
CIFAR-10_Classifier_App/
│── app/                  # Streamlit app
│── models/               # Saved models
│── reports/              # Project PDF + figures
│── data/                 # Dataset samples
│── .streamlit/           # Config for deployment
│── requirements.txt
│── README.md
```

## Tools & Skills

* **Deep Learning**: TensorFlow (Keras API), Transfer Learning
* **Deployment**: Streamlit, GitHub
* **Visualization**: Matplotlib, Seaborn
* **Workflow**: Experimentation with Colab → Fine-tuning → Deployment
