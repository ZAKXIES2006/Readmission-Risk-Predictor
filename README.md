# Readmission-Risk-Predictor
# Hospital Readmission Risk Predictor

Predicting 30-day hospital readmissions using machine learning to help hospitals identify high-risk patients and reduce costly re-admissions.

## Problem

Hospital readmissions within 30 days cost the healthcare system billions annually and indicate gaps in patient care. This project builds a predictive model that flags high-risk patients at discharge, enabling targeted follow-up interventions.

## Dataset

- **Source:** UCI Diabetes 130-Hospitals dataset (1999–2008)
- **Size:** ~100,000 patient encounters across 130 US hospitals
- **Target:** Whether a patient was readmitted within 30 days

## Project Structure



**Phase 1: ETL Pipeline** | Cleaned raw data, handled missing values, removed leakage, stored in SQLite database |
**Phase 2: EDA** | Explored readmission patterns across age groups, prior visits, medications, and hospital stay length |
**Phase 3: Predictive Model** | Trained Logistic Regression and Random Forest, compared with ROC-AUC and confusion matrices |
**Phase 4: Model Improvement** | Applied SMOTE for class balancing, added SHAP explainability for feature-level insights |
**Phase 5: Conclusions** | Summarised key risk factors and actionable recommendations for hospitals |

## Key Results

- **Best Model:** Random Forest with SMOTE balancing
- **Top Risk Factors:** Number of prior inpatient visits, discharge disposition, number of medications, number of diagnoses
- **SHAP Analysis:** Identified which features push individual patients toward higher readmission risk

## Tech Stack

- **Python** — Pandas, NumPy, Matplotlib, Seaborn
- **SQL** — SQLite for data storage and querying
- **Machine Learning** — Scikit-learn (Logistic Regression, Random Forest)
- **Class Balancing** — SMOTE (imbalanced-learn)
- **Explainability** — SHAP (TreeExplainer)

## How to Run

1. Clone the repo:
   git clone https://github.com/ZAKXIES2006/Readmission-Risk-Predictor.git
   cd Readmission-Risk-Predictor

2. Install dependencies:
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn shap ipython-sql sqlalchemy

4. Open the notebook:
   jupyter lab

5. Run all cells in `Hospital_Readmission_Predictor.ipynb`
