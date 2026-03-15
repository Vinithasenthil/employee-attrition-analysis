# Employee Attrition Analysis and Prediction

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Application](https://img.shields.io/badge/Application-Streamlit-red)
![Database](https://img.shields.io/badge/Database-SQLite-green)

---

## Table of Contents

- [Live Application](#live-application)
- [Project Overview](#project-overview)
- [Project Highlights](#project-highlights)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Project Workflow](#project-workflow)
- [Machine Learning Models](#machine-learning-models)
- [Application](#application)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Run the Application](#run-the-application)
- [Technologies Used](#technologies-used)
- [Outcome of the Project](#outcome-of-the-project)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Live Application

This project has been deployed using *Streamlit*, allowing users to interact with the employee attrition prediction system through a web interface.

You can access the application using the link below.

[Open Application](YOUR_STREAMLIT_APP_LINK)

---

## Project Overview

This project analyzes employee data to understand patterns related to *employee attrition* and builds machine learning models to *predict whether an employee is likely to leave the organization*.

The system integrates *data analysis, feature engineering, machine learning models, and an interactive application* to provide insights and predictions based on employee information.

---

## Project Highlights

- Built an end-to-end *Machine Learning project* for employee attrition prediction.
- Performed *Exploratory Data Analysis (EDA)* to identify factors influencing employee turnover.
- Applied *feature engineering techniques* to improve prediction performance.
- Trained and compared multiple *machine learning models* including Logistic Regression, Decision Tree, and Random Forest.
- Evaluated models using *Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and ROC Curve*.
- Developed an interactive *Streamlit application* for predictions and analysis.
- Implemented *bulk prediction functionality* for multiple employee records.
- Integrated *SQLite database* for storing employee data and model metrics.
- Structured the project using a *modular folder architecture*.
- Deployed the application for interactive access.

---

## Problem Statement

Employee attrition leads to increased recruitment costs, loss of experienced employees, and reduced productivity.

The objective of this project is to analyze employee data, identify patterns associated with attrition, and develop models capable of predicting whether an employee may leave the organization.

---

## Dataset

The dataset used in this project is the *IBM HR Analytics Employee Attrition dataset*.

It contains *1470 employee records* with attributes such as:

- Age
- Department
- Job Role
- Monthly Income
- Job Satisfaction
- Overtime
- Years at Company
- Performance Rating

Target variable:

*Attrition*

- Yes → Employee left the organization  
- No → Employee stayed in the organization

---

## Exploratory Data Analysis

EDA was performed to identify patterns related to employee attrition.

Visualizations used:

- Attrition Distribution
- Attrition by Department
- Attrition vs Overtime
- Income vs Attrition
- Age Distribution
- Correlation Heatmap

### Key Insights

- Around *16% of employees leave the organization*.
- *Sales department shows higher attrition* compared to others.
- Employees working *overtime have a higher likelihood of leaving*.
- *Lower income employees show higher attrition trends*.
- Younger employees show slightly *higher turnover patterns*.

---

## Feature Engineering

Additional features were created to improve model performance.

*StressRisk*  
Identifies employees working overtime who may experience higher stress.

*IncomeLevelRatio*  
Represents employee income relative to job level.

*ExperienceRate*  
Measures employee experience relative to age.

*StagnationScore*  
Indicates how long an employee has gone without promotion.

*LoyaltyScore*  
Measures stability with the current manager.

---

## Project Workflow

1. Data Collection  
2. Data Preprocessing  
3. Feature Engineering  
4. Model Training  
5. Model Evaluation  
6. Model Selection  
7. Application Integration  

---

## Machine Learning Models

The following models were trained:

- Logistic Regression
- Decision Tree
- Random Forest

### Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve

---

## Application

The application allows users to:

- Analyze employee attrition patterns
- Predict employee attrition
- Predict employee performance
- Perform bulk predictions
- View model evaluation metrics

---

## Project Structure

employee-attrition  
│  
├── app.py  
├── employee_attrition.db  
├── requirements.txt  
├── README.md  
│  
├── data  
│   └── Employee-Attrition.xlsx  
│  
├── models  
│   ├── attrition_model.pkl  
│   ├── performance_model.pkl  
│   └── scaler.pkl  
│  
├── notebooks  
│   └── employee_attrition_eda.ipynb  
│  
└── src  
    ├── database.py  
    ├── preprocess.py  
    ├── feature_engineering.py  
    ├── train.py  
    └── evaluate.py  

---

## Requirements

Python 3.10 or higher

Install dependencies:

pip install -r requirements.txt

---

## Installation

Clone the repository:

git clone https://github.com/your-username/employee-attrition-analysis.git

Move into the project directory:

cd employee-attrition-analysis

Install dependencies:

pip install -r requirements.txt

---

## Run the Application

streamlit run app.py

---

## Technologies Used

Python  
Pandas  
NumPy  
Scikit-learn  
Matplotlib  
Seaborn  
Streamlit  
SQLite  
Joblib  

---

## Outcome of the Project

This project demonstrates how employee data can be analyzed and used to predict employee attrition using machine learning techniques.

Through data analysis and predictive modeling, the system identifies important factors influencing employee turnover such as overtime, income level, and department distribution.

The project enables users to:

- Analyze employee attrition patterns
- Predict whether an employee is likely to leave
- Predict employee performance ratings
- Perform predictions for multiple employees using bulk data input

---

## Future Improvements

Possible future improvements include:

- Hyperparameter tuning
- Model explainability using SHAP or LIME
- Integration with real-time HR systems
- Automated ML pipelines

---

## Author

Vinitha S

B.Tech – Computer Science and Business Systems . 