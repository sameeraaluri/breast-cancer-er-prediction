## Breast Cancer ER Status Prediction 
## Sameera Aluri

This project predicts Estrogen Receptor (ER) status in breast cancer patients using clinical and genomic data from the METABRIC dataset. It was completed as a final project for the CISC 5800 Machine Learning course at Fordham University.

## Dataset

- **Source**: METABRIC (via Kaggle)
- https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric
- ~1,874 samples
- 600+ features: clinical data and gene mutation flags
- Target: `er_status_binary` (1 = ER+, 0 = ER−)

## Problem

Predict ER status to support treatment planning. ER+ patients are eligible for hormone therapy, making early and accessible prediction critical.

## Methods

- Data preprocessing (imputation, scaling, encoding)
- Feature reduction via `SelectKBest` (top 500 numeric features)
- Train/test split with stratification

### Models Used

- Logistic Regression (tuned with GridSearchCV)
- Support Vector Machine (linear kernel)
- Multi-layer Perceptron (MLP) — 2 hidden layers (100, 50), tuned with GridSearchCV

## Results

| Model                | Accuracy | AUC  | F1 (ER+) | F1 (ER−) |
|---------------------|----------|------|----------|----------|
| Logistic Regression | 93%      | 0.95 | 0.95     | 0.84     |
| SVM (Linear Kernel) | 93%      | 0.96 | 0.96     | 0.86     |
| MLP Neural Network  | 94%      | 0.96 | 0.96     | 0.86     |

## Feature Interpretation

Top predictive features include:
- **ER+**: GATA3, AR, cluster 4ER+
- **ER−**: TP53, EGFR  
Consistent with clinical research findings.

## Files

- `CISC_5800_Final_Project_Report_SameeraAluri.pdf` - final report in IEEE format
- `CISC_5800_Final_Project_Sameera_Aluri.ipynb` - full code and analysis (Jupyter notebook)
- `CISC_5800_Final_Project_Sameera_Aluri.pdf` - full code and analysis (PDF)
- `CISC_5800_Final_Project_Sameera_Aluri.py` - full code and analysis (Executable script)
- `METABRIC_RNA_Mutation.csv` - primary dataset


---


