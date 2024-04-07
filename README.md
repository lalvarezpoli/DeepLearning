# Predictive Modeling using Deep Learning on MIMIC-III Dataset

This project involves two predictive exercises using Deep Learning techniques: binary classification and regression. The dataset used is sourced from the [MIMIC-III (Medical Information Mart for Intensive Care III)](https://mimic.physionet.org/) project, a comprehensive database of deidentified health-related data from over forty thousand patients who stayed in critical care units at the Beth Israel Deaconess Medical Center between 2001 and 2012. Both problems utilise the same dataset for training purposes.

## Dataset
- **Source:** [MIMIC project](https://mimic.physionet.org/)
- **Description:** The MIMIC-III dataset offers a rich source of critical care information, including demographics, vital signs, laboratory tests, medications, and more.

## Predictive Exercises
1. **Binary Classification: Predicting Hospital Mortality**
   - **Objective:** Predict the probability of death during hospitalization for patients in care.
   - **Approach:** Employ deep learning techniques, namely KNN and SVM, to build a model that classifies patients into survival and non-survival groups based on various clinical features.

2. **Regression: Predicting Length of Hospital Stay**
   - **Objective:** Forecast the length of hospital stay for patients in critical care.
   - **Approach:** Utilize deep learning models, Neural Networks and Ensemling models, to predict the number of days a patient is expected to stay in the hospital based on their health records and other relevant factors.

## Implementation Details
- **Tools & Technologies:** SKlearn, Keras, Pandas.
- **Model Architectures:** Neural networks (feedforward networks)
- **Evaluation Metrics:** Accuracy, ROC-AUC (for classification); RMSE (for regression)
- **Data Preprocessing:** Feature scaling, handling missing values, one-hot + target encoding, mean imputation, etc.
- **Model Training:** Hyperparameter tuning, model training and evaluation.

## Usage
1. **Data Preprocessing:**
   - Clean and preprocess the MIMIC-III dataset to prepare it for modeling.
   - Handle missing values, normalize features, and encode categorical variables.
   - Merge extra dataset with patient comorbilities

2. **Model Building:**
   - Develop deep learning models for both classification (binary) and regression tasks.
   - Experiment with different architectures, activation functions, and optimization techniques.

3. **Model Evaluation:**
   - Evaluate model performance using cross validation metrics.
   - Fine-tune models based on validation results to achieve optimal performance.
  
4. **Predictions:**
   - Submit predictions to Kaggle competitions
      -  [Prob of Death - KNN](https://www.kaggle.com/competitions/dl24-probability-of-death-with-k-nn?rvi=1)
      -  [Prob of Death - KNN](https://www.kaggle.com/competitions/dl24-probability-of-death-with-svm?rvi=1)
      -  [Length of Stay](https://www.kaggle.com/competitions/dl24-length-of-stay-prediction-nn-ensembles?rvi=1)

## Conclusion
This project aims to demonstrate the application of deep learning in critical care analytics using real-world healthcare data. By predicting hospital mortality and length of stay accurately.
