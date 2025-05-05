# 🩺 Diabetes Progression Prediction
This project demonstrates a machine learning workflow to predict the progression of diabetes in patients using medical data. The model is trained on the Diabetes dataset from the sklearn.datasets module and aims to estimate disease progression based on key physiological attributes.

![Diabetes](https://github.com/user-attachments/assets/f560802f-29ac-4ead-9a43-fa480a6d236b)


## 🎯 Objective
To build a regression model that can predict the progression of diabetes one year after baseline measurements. This can aid healthcare professionals in early intervention and personalized patient care planning.

## 🧠 Key Features
- 📊 `Data Exploration`: Visual analysis and statistical summary of features like BMI, blood pressure, and serum measurements.

- 🔍 `Feature Engineering`: Identification of relevant predictors and normalization to improve model performance.

- 🧪 `Model Training`: Implements and evaluates various regression models including:

  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree Regressor
  - Random Forest Regressor

- 📈 `Model Evaluation`: Uses metrics like Mean Squared Error (MSE), R² Score, and cross-validation to assess model accuracy.

- 💾 `Model Persistence`: Best-performing model is saved using joblib or pickle for later inference.

- 🌐 `Web App Interface (Optional)`: Optionally integrates with Flask or Streamlit for user-friendly predictions based on new input data (can be extended).

## 🧰 Tech Stack
- `Language`: Python

- `Libraries`: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Joblib

- `ML Techniques`: Supervised Learning – Regression

- `Dataset`: Built-in Diabetes Dataset from sklearn.datasets

## 🚀 How to Run

#### Clone the repository

`git clone https://github.com/anulsasidharan/diabetes_progression.git`

`cd diabetes_progression`

#### Install dependencies

`pip install -r requirements.txt`

#### Run the notebook for training & evaluation

`jupyter notebook Diabetes_Progression_Model.ipynb`

