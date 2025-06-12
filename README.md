# IndonesiaAI_ML_Batch7_Project_02
This project focuses on building a machine learning model to detect fraudulent transactions using the fraudTrain.csv dataset. By combining data preprocessing, feature engineering, and the AdaBoost algorithm, the model learns to distinguish between normal and fraudulent patterns.

# AI Generated Summary of the JupyterNotebook Fraud Detection Project file :

This notebook implements a fraud detection pipeline using credit card transaction data and AdaBoost + RandomForest modeling.

---

## 📊 Dataset Overview

**Columns (22 total):**

- **Transaction & Timing**:  
  `trans_date_trans_time`, `unix_time`  
- **Card & User Info**:  
  `cc_num`, `first`, `last`, `gender`, `dob`, `job`  
- **Location**:  
  `street`, `city`, `state`, `zip`, `lat`, `long`  
- **Merchant Info**:  
  `merchant`, `category`, `merch_lat`, `merch_long`  
- **Transaction Details**:  
  `amt`, `city_pop`, `trans_num`  
- **Target**:  
  `is_fraud` (0 = non-fraud, 1 = fraud)

---

## 🔧 Imports & Setup

- Data handling: `pandas`, `numpy`, `datetime`  
- Plotting: `matplotlib`, `seaborn`  
- Splitting: `train_test_split`  
- Imbalance handling: `SMOTE`  
- Modeling: `AdaBoostClassifier`, `RandomForestClassifier`  
- Evaluation: `classification_report`, `confusion_matrix`, `roc_auc_score`  
- EDA reporting: ReportLab for PDF generation  
- Model persistence: `pickle`

---

## 🧭 Steps

### 1. Data Loading  
- Loaded `fraudTrain.csv`  
- (Optionally sampled 1000 rows for quick EDA)

### 2. EDA & Initial Profiling  
- Generated a PDF EDA report using `mini_eda()`—showing counts, missing data, value distributions  
- Checked null counts and dataset info

### 3. Feature Engineering

- Parsed `trans_date_trans_time` and `unix_time` into year, month, day, hour, day-of-week  
- Aligned time differences (`dow_dif`) between transaction time and record time  
- Calculated distance between cardholder and merchant using Haversine → grouped into 10 km bins  
- Computed cardholder age and grouped into 10-year brackets  
- Calculated fraud rates by: `hour`, `category`, `city`, `age grp`, `distance grp`, `dow_dif`  
- Created outlier flags based on category/job-specific 5th/95th percentiles  
- Applied one-hot encoding for `gender`

### 4. Data Splitting

- Split into train/validation/test (80/20 → then train → val 80/20)  
- Combined features and target into `df_train_8080`, `df_val_8020`, `df_test_20`

### 5. Handling Imbalance  
- Used SMOTE on `df_train_8080_selected` to balance classes  
- Verified balanced distribution

### 6. Feature Selection

- Initial feature set manually defined  
- Assessed correlations with a heatmap  
- After modeling, identified important features (≥1% importance)

### 7. Modeling: AdaBoost + RandomForest

**Model A (all selected features):**
- Trained on SMOTE-balanced data  
- Evaluated on validation and test sets: classification report, confusion matrix, ROC-AUC  
- Saved model as pickle

**Model B (top features only):**
- Re-trained using only features with ≥1% importance  
- Evaluated identically  
- Saved separately

---

## ✅ Summary Notes

- Complete pipeline from raw CSV to evaluated model  
- Feature engineering rich in time, location, demographic, and outlier-based flags  
- Two modeling rounds: full vs. trimmed feature sets  
- Model evaluation metrics captured and models saved for deployment/testing

---

