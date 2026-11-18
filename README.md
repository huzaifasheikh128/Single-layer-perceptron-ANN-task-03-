# Single-Layer Perceptron from Scratch (ANN Lab Task 3)

## 📌 Overview
This repository contains the implementation of a **Single-Layer Perceptron** built entirely from scratch using **Python** and **NumPy**—without relying on prebuilt machine learning libraries (such as `scikit-learn`, `TensorFlow`, or `PyTorch`). 

The model is trained and evaluated on a publicly available binary classification dataset as part of the **Artificial Neural Networks (ANN) Lab**.

---

## 🎯 Objectives
- Select and load a real-world binary classification dataset from a public repository (UCI Machine Learning Repository / Kaggle).
- Perform data preprocessing (label encoding, missing value handling, train-test splitting, and feature scaling).
- Implement the Perceptron learning algorithm and step activation function from scratch using NumPy.
- Evaluate model performance using standard classification metrics (Accuracy, Precision, Recall, F1-Score, and Confusion Matrix).

---

## 📊 Dataset
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Data Set *
- **Source:** [UCI Machine Learning Repository](https://uci.edu)
- **Instances:** 569 samples
- **Features:** 30 numerical features computed from digitized images of breast mass aspirates
- **Target Classes:** Binary (`0` = Benign, `1` = Malignant)

---

## ⚙️ Methodology & Implementation
1. **Data Preprocessing:**
   - Encoded categorical target labels into binary values (`0` and `1`).
   - Shuffled and split the dataset into **80% training** and **20% testing** sets.
   - Applied **Z-score standardization** using training set statistics to prevent data leakage.
2. **Model Architecture:**
   - **Linear Combination:** $z = \mathbf{w} \cdot \mathbf{x} + b$
   - **Activation Function:** Binary Step Function ($\hat{y} = 1$ if $z \geq 0$, else $0$)
   - **Weight Update Rule:** $\mathbf{w} \leftarrow \mathbf{w} + \eta (y - \hat{y})\mathbf{x}$ and $b \leftarrow b + \eta (y - \hat{y})$
3. **Evaluation Metrics (Implemented from Scratch):**
   - Confusion Matrix (TP, TN, FP, FN)
   - Accuracy, Precision, Recall, and F1-Score

---

## 🚀 How to Run

### Prerequisites
- Python 3.8+
- NumPy

### Installation & Execution
1. Clone the repository:
   ```bash
   git clone https://github.com<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```
2. Install NumPy (if not already installed):
   ```bash
   pip install numpy
   ```
3. Run the script:
   ```bash
   python perceptron.py
   ```
   *(Replace `perceptron.py` with your actual script or `.ipynb` filename.)*

---

## 📈 Sample Results

| Metric | Score |
| :--- | :---: |
| **Accuracy** | 96.49% |
| **Precision** | 97.62% |
| **Recall** | 93.18% |
| **F1-Score** | 95.35% |

---

## 🛠️ Tech Stack
- **Language:** Python
- **Core Library:** NumPy
