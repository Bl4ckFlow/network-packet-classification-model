📘 Intrusion Detection System on UNSW-NB15

A complete Machine Learning + Deep Learning pipeline

This project implements a full Intrusion Detection System (IDS) using the UNSW-NB15 dataset.
It includes preprocessing, classical ML models, ensemble methods, and deep learning architectures for detecting malicious network traffic.

🚀 Features

✔ Complete Data Pipeline

    Load training & testing sets
    
    Cleaning & handling missing data
    
    Encoding categorical variables
    
    Correlation-based feature reduction
    
    Standardization with StandardScaler

✔ Machine Learning Models

    Baseline Random Forest
    
    Optimized Random Forest via Grid Search
    
    Bagging Ensemble (RF as base estimator)
    
    Voting Ensemble (RF + Logistic Regression + SVM)

✔ Deep Learning Models

    MLP Neural Network
    
    LSTM Classifier
    
    Autoencoder (Anomaly Detection, ROC-optimized)

✔ Model Evaluation

    Accuracy, Precision, Recall, F1-score
    
    Confusion matrices
    
    Feature importance plots
    
    Comparison tables for ML & DL models
    
    ROC-based threshold tuning for Autoencoder


📊 Dataset: UNSW-NB15

    The UNSW-NB15 dataset contains realistic modern attack traffic, including:
    
    257,673 samples
    
    45 raw features
    
    10 attack categories
    
    Binary label: Normal (0) vs Attack (1)
    
    More information:
    https://research.unsw.edu.au/projects/unsw-nb15-dataset




🤖 Models Implemented

  1. Baseline Random Forest

    Solid starting performance (~92% accuracy).

  2. Optimized Random Forest

    Grid Search on:
  
      n_estimators
      
      max_depth
      
      min_samples_split
      
      min_samples_leaf
      
      bootstrap
      
      Often the best ML model.

  3. Bagging Ensemble

    Uses the optimized RF as the base estimator.
    Improves variance & stability.

  4. Voting Ensemble

    Soft voting using:

    Random Forest
    
    Logistic Regression
    
    SVM (RBF kernel)


🤖 Deep Learning Models
  1. MLP Neural Network

    Dense layers + dropout
    
    ~93.8% F1-score
    
    Excellent supervised performance

  2. LSTM Classifier

    Sequence-based traffic modeling
    
    Performs similarly to MLP

  3. Autoencoder (Unsupervised)

    Trained only on normal traffic
    
    Detects anomalies via reconstruction error
    
    Threshold optimized via ROC + Youden’s J statistic
    
    ~80% F1-score (unsupervised!)


🥇 Model Comparison Summary
| Model            | Accuracy  | Precision | Recall | F1-score  |
| ---------------- | --------- | --------- | ------ | --------- |
| **Optimized RF** | ~0.94     | ~0.94     | ~0.95  | **0.94+** |
| **MLP**          | 0.92      | 0.93      | 0.94   | 0.94      |
| **LSTM**         | 0.92      | 0.91      | 0.95   | 0.93      |
| **Autoencoder**  | 0.78–0.80 | 0.92      | 0.71   | 0.80      |



