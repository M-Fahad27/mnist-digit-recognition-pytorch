# Handwritten Digit Recognition using PyTorch

A deep learning project that trains a **Neural Network in PyTorch** to recognize handwritten digits from the **MNIST dataset**, achieving **~97% test accuracy** and successfully predicting custom handwritten digit images.

---

## 📌 Project Overview

This project demonstrates a complete **end-to-end deep learning workflow**:

- Loading and preprocessing the MNIST dataset  
- Building a **fully connected neural network** in PyTorch  
- Training and evaluating the model  
- Saving the trained model  
- Predicting digits from a **real handwritten image** using OpenCV  

---

## 🧠 Model Architecture

**Input:**
- 28 × 28 grayscale image → flattened to **784 features**

**Fully Connected Layers:**
- 784 → 128 neurons  
- 128 → 64 neurons  
- 64 → 10 output classes (digits 0–9)

**Activation Function:**
- **ReLU**  
  - Fast convergence  
  - Helps prevent vanishing gradient problem  

**Loss Function:**
- **CrossEntropyLoss** → best for multi-class classification  

**Optimizer:**
- **Adam (learning rate = 0.001)**  
  - Adaptive learning  
  - Efficient and stable training  

---

## 📊 Results

- **Test Accuracy:** ~97%  
- Successfully predicts **custom handwritten digit images** after preprocessing:
  - Grayscale conversion  
  - Thresholding  
  - Resizing to **28×28**  
  - Normalization  

---

## 📂 Project Structure

```text
.
├── train.py           # Training, evaluation, and saving the model
├── test.py            # Script to load the model and predict digits from custom images
├── mnist_model.pth    # Saved state dictionary (trained weights)
├── test_digit.png     # Sample handwritten digit for testing
└── README.md          # Project documentation and setup guide
```

---

## ⚙️ Installation

```bash
git clone https://github.com/M-Fahad27/mnist-digit-recognition-pytorch
cd mnist-digit-recognition-pytorch
pip install torch torchvision opencv-python numpy
```

## 🚀 Training the Model

```bash
python train.py
```
This will:

- Train the neural network
- Display epoch loss and test accuracy
- Save the trained model as mnist_model.pth

---

## 🔍 Testing on Custom Image

- Place your handwritten digit image as:

```bash
test_digit.png
```
- Run:

```bash
python test.py
```
- Output Example:

```bash
Predicted Digit: 7
```
---

## 🎯 Key Learnings

- Importance of data normalization and preprocessing
- Understanding neural network architecture design
- Role of activation functions, optimizers, and loss functions
- Complete training → evaluation → inference pipeline in PyTorch

---

## 🔮 Future Improvements

- Implement a CNN model for higher accuracy
- Build a Flask/FastAPI web app for real-time prediction
- Deploy on AWS or cloud platform

---

## 👨‍💻 Author

Muhammad Fahad
Aspiring AI Engineer | Machine Learning | Deep Learning

---




