# AI Healthcare – Intelligent Disease Prediction

## 📌 Project Overview

AI Healthcare – Intelligent Disease Prediction is a machine learning based web application designed to provide preliminary disease risk predictions using patient health information.

The system provides separate prediction modules for:

- 🩸 Diabetes
- ❤️ Heart Disease
- 🎗️ Breast Cancer

The application is developed using Python and Flask. Trained machine learning models are integrated into a web interface where users can enter health-related information and receive a prediction along with a confidence value, health suggestions, and a downloadable PDF report.

> **Disclaimer:** This application is developed for educational and project purposes only. It is not a substitute for professional medical diagnosis or treatment.

---

## 🎯 Objectives

- Develop a web-based healthcare prediction application.
- Integrate trained machine learning models with Flask.
- Provide separate prediction modules for different diseases.
- Display prediction results and confidence values.
- Provide health suggestions based on the prediction.
- Maintain recent prediction history.
- Generate downloadable PDF prediction reports.
- Provide a simple and user-friendly interface.

---

## 🩺 Diseases Covered

### 1. Diabetes Prediction

The Diabetes module accepts patient information such as:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

### 2. Heart Disease Prediction

The Heart Disease module uses health parameters including:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- ST Depression
- Slope
- Number of Major Vessels
- Thalassemia

### 3. Breast Cancer Prediction

The Breast Cancer module uses numerical measurements derived from breast cancer diagnostic data.

The features are organized into:

- Mean measurements
- Standard error measurements
- Worst measurements

---

## ✨ Features

- 🩸 Diabetes prediction
- ❤️ Heart disease prediction
- 🎗️ Breast cancer prediction
- 📊 Prediction confidence
- 🟢 Low-risk / 🔴 High-risk result display
- 💡 Health suggestions
- 📋 Recent prediction history
- 📄 Downloadable PDF reports
- 🕒 Prediction date and time
- 📱 Responsive web interface
- ⚠️ Medical disclaimer

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Backend
- Flask

### Machine Learning
- Scikit-learn
- NumPy
- Joblib

### Frontend
- HTML
- CSS
- JavaScript

### Reporting
- ReportLab

### Development Tools
- Visual Studio Code
- Jupyter Notebook
- Git
- GitHub

---

## 📂 Project Structure

```text
AI-Healthcare-Disease-Prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── datasets/
│   ├── breast_cancer.csv
│   ├── diabetes.csv
│   └── heart_disease_uci.csv
│
├── models/
│   ├── cancer_model.pkl
│   ├── cancer_scaler.pkl
│   ├── diabetes_model.pkl
│   ├── heart_model.pkl
│   ├── heart_scaler.pkl
│   └── scaler.pkl
│
├── notebooks/
│   ├── cancer_training.ipynb
│   ├── diabetes_training.ipynb
│   └── heart_training.ipynb
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── images/
│   └── js/
│
└── templates/
    ├── index.html
    ├── diabetes.html
    ├── heart.html
    └── cancer.html
