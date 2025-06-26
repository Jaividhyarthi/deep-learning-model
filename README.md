# Task 2 – Deep Learning Model: Image Classification (MNIST)

This repository contains Task 2 of my Data Science Internship at CODTECH.

## ✅ Objective

To build and train a **Convolutional Neural Network (CNN)** using **TensorFlow/Keras** to classify handwritten digits from the **MNIST dataset**. The task includes training, evaluating, and visualizing the performance of the model.

---

## 🧪 Tools & Libraries

- Python
- TensorFlow / Keras
- Matplotlib

---

## 📊 Dataset Used

**MNIST Dataset**  
- 60,000 training images  
- 10,000 test images  
- Each image is a 28×28 grayscale digit (0–9)

---

## 🧠 Model Architecture

- **Conv2D** (32 filters, 3×3) → ReLU
- **MaxPooling2D** (2×2)
- **Conv2D** (64 filters, 3×3) → ReLU
- **MaxPooling2D** (2×2)
- **Flatten**
- **Dense Layer** (64 units) → ReLU
- **Output Layer** (10 units, Softmax)

---

## 🧮 Training Details

- **Loss Function:** Sparse Categorical Crossentropy
- **Optimizer:** Adam
- **Epochs:** 5
- **Validation:** 10,000 test images

---

## 📈 Visualizations

After training, the following graphs are generated:
- **Accuracy vs Epochs**
- **Loss vs Epochs**

These are saved as `training_metrics.png` in the output.

---

## 🚀 How to Run

Ensure TensorFlow is installed:
```bash
pip install tensorflow
Task 2 - Deep Learning Model/
├── mnist_cnn_model.py
├── training_metrics.png
├── README.md
