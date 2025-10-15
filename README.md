# 🩺 Breast Cancer Prediction Model

This project aims to predict whether a breast tumor is **malignant** or **benign** using various **machine learning algorithms**. It demonstrates the power of data-driven models in assisting early diagnosis and improving healthcare decision-making.

---

## 📘 Overview

The project compares the performance of five algorithms:
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Random Forest  
- Support Vector Machine (SVM)  
- Neural Network  

Each model was trained and evaluated on a **publicly available dataset** to identify the most accurate and reliable approach for breast cancer classification.

---

## 🧠 Methodology

1. **Data Preprocessing**
   - Removed irrelevant columns (`id`, `Unnamed: 32`)
   - Checked for null values and handled missing data
   - Standardized numerical features for scaling-sensitive models

2. **Model Implementation**
   - Trained Logistic Regression, KNN, Random Forest, SVM, and Neural Network
   - Used **Scikit-learn** for classical ML models and **TensorFlow/Keras** for deep learning
   - Evaluated models using metrics such as **Accuracy**, **Precision**, **Recall**, and **F1-score**

3. **Performance Evaluation**
   - Compared results using a **Confusion Matrix**
   - Determined top-performing models based on predictive accuracy and computational efficiency

---

## ⚙️ Tools & Libraries

- **Language:** Python  
- **Libraries Used:**
  - `pandas`, `numpy` – for data handling and preprocessing  
  - `scikit-learn` – for implementing ML models and evaluation metrics  
  - `tensorflow`, `keras` – for neural network model  
  - `matplotlib`, `seaborn` – for visualization  

---

## 🚀 Steps to Run the Project

```bash
# Step 1: Clone the repository
git clone https://github.com/yourusername/Breast-Cancer-Prediction.git
cd Breast-Cancer-Prediction

# Step 2: Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # for Windows
source venv/bin/activate  # for Mac/Linux

# Step 3: Install dependencies manually
pip install pandas numpy scikit-learn tensorflow keras matplotlib seaborn

# Step 4: Run the project
python breast_cancer_prediction.py
