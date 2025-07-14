Data Preprocessing and EDA** and (Machine Learning & Deep Learning Models)** of Employee Attrition & Performance project:

# Employee Attrition and Performance Prediction

## 📊 Overview
This project analyzes employee data to uncover patterns behind attrition and predict performance scores using machine learning (Random Forest & Linear Regression) and deep learning (Neural Network). It includes data preprocessing, exploratory analysis, statistical testing, classification, regression, and model evaluation.

## 📁 Dataset Description

**File:** `employee_data.csv`  
**Shape:** 100 rows × 8 columns  
**Columns:**
- `EmployeeID`: Unique identifier
- `Name`: Employee name
- `Age`: Employee age
- `Department`: Department name (e.g., HR, Sales)
- `Salary`: Annual salary (numeric)
- `YearsAtCompany`: Total years spent at the company
- `PerformanceScore`: Annual performance score (out of 100)
- `Attrition`: Whether the employee left the company (Yes/No)

## ✅ Step 1: Data Cleaning & Preprocessing

- **Missing Values**: Filled numeric columns using column mean
- **Duplicates**: Removed duplicate records
- **Inconsistent Entries**: Cleaned and standardized `Department` values
- **Descriptive Stats**: Summary generated using `df.describe()`

## 📊 Step 2: Exploratory Data Analysis (EDA)

### Visualizations:
- **Pairplot**: To observe attrition patterns across features
- **Correlation Heatmap**: To assess feature relationships
- **Boxplot**: To detect outliers in `Age`, `Salary`, `YearsAtCompany`
- **Attrition Probability by Department**:
  - Sales: 35.89%
  - Engineering: 30.77%
  - HR: 23.08%
  - Marketing: 10.26%

### Statistical Tests:
- **Bayesian Inference**:
  - P(Attrition | PerformanceScore) ≈ 0.395
- **ANOVA (F-test)**:
  - Significant difference in performance scores across departments (p-value ≈ 2.56e-12)

## 🤖 Step 3: Machine Learning Models

### Random Forest Classifier
- **Target**: `Attrition`
- **Accuracy**: 70%
- **Precision/Recall (Class 1)**: 57%
- **Confusion Matrix**: Visualized using Seaborn heatmap

### Linear Regression
- **Target**: `PerformanceScore`
- **R² Score**: 0.74
- **Cross-Validated R² (5-fold)**: 0.73
- **MAE**: 0.36  
- **Visuals**:
  - Predicted vs Actual
  - Residuals Plot

---

## 🧠 Step 4: Deep Learning - Neural Network (Keras)

- **Architecture**:
  - Dense(64, ReLU)
  - Dense(32, ReLU)
  - Dense(1, Linear)
- **Loss**: MSE  
- **Final Validation MAE**: ~0.33  
- **Test MAE**: ~0.49

---

## 📈 Trend Analysis

- **Performance vs Tenure**: Performance generally increases with `YearsAtCompany`
- **Performance by Department & Attrition**: Insights into department-specific attrition-performance patterns

---

## 🔧 Tools & Libraries Used

- **Python Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`, `keras`
- **ML Models**: Random Forest, Linear Regression
- **DL Model**: Keras Sequential Neural Network

---

## 📌 Future Improvements

- Hyperparameter tuning (Random Forest & Neural Net)
- Feature engineering with interaction terms
- Deployment using Streamlit or Flask
- SHAP/LIME for model explainability




