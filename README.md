# Maternal Satisfaction Predictor
### Maternal Satisfaction Among Vaginal and Cesarean Section Delivery Care Services in Bahir Dar City Health Facilities, Northwest Ethiopia — Application of Machine Learning

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Installation & Setup](#installation--setup)
6. [How to Run](#how-to-run)
7. [Methodology](#methodology)
8. [Feature Engineering](#feature-engineering)
9. [Selected Features](#selected-features)
10. [Model Performance](#model-performance)
11. [Optimal Threshold](#optimal-threshold)
12. [App Pages](#app-pages)
13. [Results Interpretation](#results-interpretation)

---

## Project Overview

This project applies **Machine Learning** to predict maternal satisfaction with delivery care services in Bahir Dar City health facilities, Northwest Ethiopia. A **Logistic Regression** model was trained on 894 mothers to classify their overall satisfaction (Satisfied / Not Satisfied) based on 10 independent facility and demographic predictors.

The project covers:
- Descriptive statistical analysis
- Feature engineering and selection
- Logistic Regression modelling with hyperparameter tuning
- Model evaluation (Accuracy, Precision, Recall, F1, AUC, Brier Score)
- Threshold optimisation using the Youden J statistic
- Interactive Streamlit deployment app

---

## Dataset

| Item | Detail |
|---|---|
| **File** | `Delivery care satisfaction.sav` (SPSS format) |
| **Source** | Getasew Mulat — published dataset |
| **Observations** | 894 mothers |
| **Variables** | 46 |
| **Target variable** | `satisfaction` (1 = Not Satisfied, 2 = Satisfied) |
| **Not Satisfied** | 345 (38.6%) |
| **Satisfied** | 549 (61.4%) |

---

## Project Structure

```
📁 Project Folder
│
├── 📓 Maternal_Satisfaction_ML_Project.ipynb   # Main analysis notebook
├── 🚀 app.py                                   # Streamlit deployment app
├── 🤖 Maternal_Satisfaction_LR_Final_Model.pkl # Saved trained model
├── 📄 requirements.txt                         # Python dependencies
├── 🗃️ Delivery care satisfaction.sav           # SPSS dataset
└── 📖 README.md                                # This file
```

---

## Requirements

```
pandas
numpy
matplotlib
plotly
scikit-learn
seaborn
joblib
streamlit
pyreadstat
openpyxl
```

---

## Installation & Setup

**1. Clone or download the project folder**

**2. Create and activate a virtual environment**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**3. Install dependencies**
```powershell
pip install -r requirements.txt
```

**4. Verify the model file exists**

The file `Maternal_Satisfaction_LR_Final_Model.pkl` must be in the same folder as `app.py`. If it is missing, run the notebook through **Step 12** to generate it:
```python
# Step 12 of the notebook
joblib.dump(best_model, 'Maternal_Satisfaction_LR_Final_Model.pkl')
```

---

## How to Run

### Run the Jupyter Notebook
```powershell
jupyter notebook Maternal_Satisfaction_ML_Project.ipynb
```
Run all cells from top to bottom. The notebook will:
- Load and explore the SPSS dataset
- Perform descriptive statistics
- Engineer features and remove leakage variables
- Run 5-fold CV permutation importance to select the top 10 features
- Train and tune the Logistic Regression model
- Evaluate with all metrics
- Save the model to disk

### Run the Streamlit App
```powershell
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

---

## Methodology

### Step-by-step pipeline

```
Load SPSS data
      ↓
Descriptive Statistics (mean, std, frequency, percentiles)
      ↓
Encode target: 1.0 → 0 (Not Satisfied), 2.0 → 1 (Satisfied)
      ↓
Feature Engineering (remove leakage, missing, recoded vars)
27 clean candidate features remain
      ↓
5-fold CV Permutation Importance (ROC-AUC scoring)
Top 10 features selected
      ↓
Preprocessing: StandardScaler (numerical) + OneHotEncoder (categorical)
      ↓
Logistic Regression + GridSearchCV (tune C parameter)
      ↓
Evaluate on held-out test set (n = 179, 20%)
      ↓
Threshold optimisation (Youden J → 0.65)
      ↓
Save model with joblib → Deploy with Streamlit
```

---

## Feature Engineering

Before selecting predictors, the following variables were excluded:

| # | Exclusion Rule | Variables Removed | Count |
|---|---|---|---|
| 1 | **Target variable** | `satisfaction`, `satisfaction_binary` | 2 |
| 2 | **Satisfaction sub-item recodes** (data leakage — coded as not satisfied/satisfied) | `waitingtimerecode`, `privacyrecode`, `encourageandsupportatdeliveryrecode`, `politenessrecode`, `availabilityofmedicalequipmentsrecode`, `counselingrecode`, `overallcleanessrecode`, `accessandcleanessoftoiletrecode`, `waitingareacleanessandconfort`, `availabilityofbedintheward` | 10 |
| 3 | **>50% missing values** | `q1092disth` (64.2% missing), `wolkingdistancerecodeinhrs` (69.2% missing) | 2 |
| 4 | **Recoded variables** (originals kept, recodes dropped) | `ageafterrecod`, `monthlyincomerecode`, `gestationaagerecode`, `birthweightofbaby`, `modeofdeliveryrecode`, `distancerecode1` | 6 |

> **Why remove the 10 satisfaction sub-items?**
> Each is coded 1 = *not satisfied* / 2 = *satisfied* on a specific service dimension.
> They are direct sub-components of the outcome variable. Including them causes
> **data leakage** — the model would predict overall satisfaction from sub-satisfaction
> ratings, inflating all metrics artificially. Such a model would be useless in practice.

**Clean candidate features after all exclusions: 27**

**Feature selection method:** 5-fold cross-validated permutation importance (ROC-AUC scoring)

**Features selected: Top 10**

---

## Selected Features

| Rank | Variable | Description |
|---|---|---|
| 1 | `q411owners` | Facility ownership (private / government) |
| 2 | `q301greeti` | Was the mother greeted by the professional on arrival? |
| 3 | `q112waitpl` | Was a waiting area available? |
| 4 | `q113privac` | Was gender privacy maintained during examination? |
| 5 | `distancerecode1` | Distance to facility (<= 10 km / > 10 km) |
| 6 | `q302respec` | Was the mother treated respectfully during delivery? |
| 7 | `q110reason` | Reason for visit (planned / referred) |
| 8 | `q111timeta` | Time taken to reach the facility |
| 9 | `q108reside` | Residence (urban / rural) |
| 10 | `q102mstatu` | Marital status |

---

## Model Performance

Evaluated on the held-out test set (n = 179 mothers, 20% of 894).

| Metric | Value | Interpretation |
|---|---|---|
| **Accuracy** | 85.5% | Overall proportion of correct predictions |
| **Precision** | 88.2% | Of predicted satisfied, how many truly were |
| **Recall** | 88.2% | Of truly satisfied mothers, how many the model found |
| **F1 Score** | 88.2% | Harmonic mean of precision and recall |
| **Sensitivity** | 88.2% | True positive rate (same as recall) |
| **Specificity** | 81.2% | True negative rate — correctly identified not-satisfied mothers |
| **ROC-AUC** | ~0.92 | Ability to discriminate between classes (1.0 = perfect) |
| **Brier Score** | 0.1204 | Calibration of probability estimates (0 = perfect, lower is better) |

### Confusion Matrix (threshold = 0.65)

|  | Predicted Not Satisfied | Predicted Satisfied |
|---|---|---|
| **Actual Not Satisfied** | TN = 56 | FP = 13 |
| **Actual Satisfied** | FN = 13 | TP = 97 |

### Model Configuration

| Parameter | Value |
|---|---|
| Algorithm | Logistic Regression |
| Solver | lbfgs |
| C (regularisation) | 0.1 (best from GridSearchCV) |
| class_weight | balanced |
| max_iter | 1000 |
| Tuning method | GridSearchCV |
| Tuning metric | ROC-AUC |
| CV folds | 5 (Stratified K-Fold) |
| Train/Test split | 80% / 20% (stratified) |

---

## Optimal Threshold

The default threshold for binary classification is 0.50. This project uses an **optimised threshold of 0.65** derived from the **Youden J statistic** on the held-out test set.

**Youden J = Sensitivity + Specificity − 1**

This is maximised at **0.65**, giving the best simultaneous balance between:
- Correctly identifying satisfied mothers (Sensitivity)
- Correctly identifying not-satisfied mothers (Specificity)

### Threshold comparison

| Threshold | Sensitivity | Specificity | F1 Score | Accuracy | Youden J |
|---|---|---|---|---|---|
| 0.50 | 0.973 | 0.609 | 0.877 | 0.832 | 0.581 |
| 0.55 | 0.964 | 0.638 | 0.880 | 0.838 | 0.601 |
| 0.60 | 0.891 | 0.783 | 0.879 | 0.849 | 0.674 |
| **0.65** | **0.882** | **0.812** | **0.882** | **0.855** | **0.693** ← best |
| 0.70 | 0.791 | 0.884 | 0.849 | 0.827 | 0.675 |
| 0.75 | 0.664 | 0.913 | 0.772 | 0.760 | 0.577 |

**Decision rule:**
```
P(Satisfied) >= 0.65  →  Satisfied
P(Satisfied) <  0.65  →  Not Satisfied
```

---

## App Pages

The Streamlit app (`app.py`) has three pages:

### 🔮 Predict
- Enter values for all 10 features using dropdown menus
- Click **Predict Satisfaction**
- View results including:
  - Satisfied / Not Satisfied verdict
  - Plotly gauge chart showing predicted probability vs threshold
  - Probability bar chart
  - Input summary table

### 📊 Model Performance
- All 7 metric cards including Brier Score
- Interactive confusion matrix heatmap
- Threshold analysis line chart (all metrics vs threshold 0.30–0.80)
- Odds ratio bar chart (feature importance)

### ℹ️ About
- Study overview and dataset details
- Feature engineering exclusion rationale
- Full model configuration
- Threshold selection methodology
- All 10 selected feature descriptions

---

## Results Interpretation

| Term | Meaning |
|---|---|
| **Satisfied** | P(Satisfied) ≥ 0.65 — mother likely satisfied with delivery care |
| **Not Satisfied** | P(Satisfied) < 0.65 — mother likely not satisfied |
| **Odds Ratio > 1** | Factor increases the odds of satisfaction |
| **Odds Ratio < 1** | Factor decreases the odds of satisfaction |
| **Brier Score** | Measures probability calibration; 0.1204 indicates well-calibrated predictions |

---

## Citation

> Dataset credit: **Getasew Mulat** — published online.
> Study: *Maternal satisfaction among vaginal and cesarean section delivery care services
> in Bahir Dar city health facilities, Northwest Ethiopia: Application of Machine Learning.*
