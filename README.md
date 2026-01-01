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

The models were evaluated primarily on **Recall** (to minimize False Negatives). The table below summarizes the test set performance for the best-performing iteration of each model.

### 📈 Model Performance Comparison (Sorted by Recall then Precision)
---

| model_name | accuracy | precision | recall | f1_score |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression (Class Reweighting) | 0.375570 | 0.181818 | 0.886228 | 0.301733 |
| ElasticNet (Class Reweighting) | 0.273473 | 0.159827 | 0.886228 | 0.270814 |
| Ridge (Class Reweighting) | 0.449407 | 0.196106 | 0.844311 | 0.318284 |
| Logistic Regression (Undersampling) | 0.513218 | 0.202593 | 0.748503 | 0.318878 |
| Logistic Regression (SMOTE) | 0.577940 | 0.221805 | 0.706587 | 0.337625 |
| ElasticNet (SMOTE) | 0.670009 | 0.265060 | 0.658683 | 0.378007 |
| Ridge (Undersampling) | 0.671832 | 0.265207 | 0.652695 | 0.377163 |
| Ridge (SMOTE) | 0.663628 | 0.259524 | 0.652695 | 0.371380 |
| ElasticNet (Undersampling) | 0.670009 | 0.261614 | 0.640719 | 0.371528 |
| KNN (Undersampling) | 0.650866 | 0.235294 | 0.574850 | 0.333913 |
| KNN (SMOTE) | 0.614403 | 0.207763 | 0.544910 | 0.300826 |

### Conclusion

The **Logistic Regression (Class Reweighting)** model achieved the highest **Recall** of **0.886228**, meaning it correctly identified 88% of the actual CHD risk cases in the test set. For a real-world medical screening tool where **False Negatives are life-critical**, the high recall of the Class Reweighting models is preferable, despite the higher number of false alarms (low precision).


