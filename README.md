# Data-science-project
# 🩺 Heart Disease Prediction using the Framingham Dataset

## 📝 Project Overview

This project focuses on predicting the **10-year risk of Coronary Heart Disease (CHD)** using the Framingham Heart Study dataset. Cardiovascular diseases are a major global health concern, and early prognosis is crucial for lifestyle intervention and reducing complications.

The primary objective is to develop a robust classification model that can accurately identify high-risk patients. Given the life-critical nature of the prediction, the modeling approach prioritizes maximizing **Recall** (correctly identifying positive CHD cases) while maintaining acceptable Precision.

---

## 💾 About the Dataset

The dataset is derived from the **Framingham Heart Study**, an ongoing cardiovascular study in Massachusetts. It is publicly available on Kaggle.

| Detail | Value |
| :--- | :--- |
| **Source** | [Kaggle - HEART DISEASE PREDICTION](https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression/data) |
| **Records** | Over 4,000 (3,656 after cleaning) |
| **Attributes** | 15 clinical and demographic features |
| **Target Variable** | `ten_year_chd` (Binary: 1 = Yes, 0 = No) |

### Key Attributes:

The features include demographic (Sex, Age), behavioral (Current Smoker, Cigs Per Day), and medical factors (BP Meds, Prevalent Stroke, Diabetes, Total Chol, Sys BP, Dia BP, BMI, Heart Rate, Glucose).

### Target Distribution (Class Imbalance)

The target variable exhibits a significant class imbalance:
* **Class 0 (No CHD Risk):** 3,099 cases
* **Class 1 (CHD Risk):** 557 cases
* **Ratio:** Approximately **5.6:1**

This imbalance necessitates the use of specialized techniques (Class Reweighting, SMOTE, Undersampling) to prevent the model from becoming biased towards the majority class.

---

## 🧹 Data Cleaning and Preprocessing

### 1. Missing Value Handling
* Missing values, primarily in the `glucose` attribute, were handled by **dropping the corresponding rows** (`df.dropna(inplace=True)`).

### 2. Data Type Conversion
* The data types for `education`, `cig_per_day`, and `bp_meds` were explicitly converted to `int64` for consistency.

### 3. Feature Selection
Based on Exploratory Data Analysis (EDA) and domain understanding:
* **Removed:** `education` (deemed unrelated to core health parameters)
* **Removed:** `current_smoker` (due to high multicollinearity with `cig_per_day`)

### 4. Scaling
* The numerical features (`age`, `cig_per_day`, `total_chol`, `sys_bp`, `dia_bp`, `bmi`, `heart_rate`, `glucose`) were scaled using **RobustScaler** for most models to mitigate the influence of outliers (particularly in skewed variables like `glucose`).
* **MinMaxScaler** was used specifically for the **K-Nearest Neighbors (KNN)** model due to its sensitivity to feature scale.

### 5. Data Splitting
* The data was split into training and testing sets with a ratio of **70:30** (`test_size=0.3`), using **stratification** on the target variable (`stratify=y`) to ensure both sets maintained the original class balance.

---

## 🧠 Modelling Approach

The project rigorously tested multiple classification algorithms combined with different imbalance handling techniques, optimizing for **Recall**. **GridSearchCV** with 5-fold cross-validation was used, setting the `refit` parameter to **'recall'** to prioritize models that best identify positive CHD cases.

### Imbalance Handling Techniques Applied:

1.  **Class Reweighting:** Adjusting the penalty for misclassifying the minority class (CHD Risk) during model training.
2.  **Oversampling (SMOTE):** Synthetic Minority Over-sampling Technique was used to create synthetic samples for the minority class.
3.  **Undersampling (Random UnderSampler):** Randomly removing samples from the majority class to balance the training data.

### Algorithms Tested:

| Algorithm | Type | Imbalance Techniques Tested | Scaling |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Linear Classifier | Reweighting, SMOTE, Undersampling | RobustScaler |
| **ElasticNet (Logistic)** | Regularized Linear Model | Reweighting, SMOTE, Undersampling | RobustScaler |
| **K-Nearest Neighbors (KNN)** | Instance-based | SMOTE, Undersampling | MinMaxScaler |
| **Ridge Classifier** | Linear Classifier | Reweighting, Undersampling | RobustScaler |

---

## 🏆 Model Results

The models were evaluated primarily on **Recall** (to minimize False Negatives) and **F1-Score** (the harmonic mean of Precision and Recall). The table below summarizes the test set performance for the best-performing iteration of each model.

| Model & Technique | Accuracy | Precision | **Recall** | **F1-Score** |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Undersampling)** | 0.69 | 0.20 | **0.62** | 0.30 |
| **ElasticNet (Undersampling)** | 0.69 | 0.20 | **0.61** | 0.30 |
| **Ridge Classifier (Reweighting)** | 0.81 | 0.33 | 0.49 | 0.39 |
| **Logistic Regression (Reweighting)** | 0.82 | 0.33 | 0.44 | 0.38 |
| ElasticNet (Reweighting) | 0.82 | 0.32 | 0.42 | 0.36 |
| KNN (SMOTE) | 0.77 | 0.24 | 0.45 | 0.32 |
| KNN (Undersampling) | 0.70 | 0.21 | 0.60 | 0.31 |

### Conclusion

The **Logistic Regression (Undersampling)** model achieved the highest **Recall** of **0.62**, meaning it correctly identified 62% of the actual CHD risk cases in the test set.

While Undersampling and SMOTE boosted recall, they significantly lowered precision (high False Positives). The **Ridge Classifier (Class Reweighting)** provided the best balance between recall and precision, yielding a Recall of **0.49** and the highest **F1-Score of 0.39**.

For a real-world medical screening tool where **False Negatives are life-critical**, the high recall of the Undersampling models is preferable, despite the higher number of false alarms (low precision).


