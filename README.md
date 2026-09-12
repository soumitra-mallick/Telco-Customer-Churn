# Telco Customer Churn Prediction

Predicting customer churn for a telecom company using classification 
models. Built as part of my data science portfolio to demonstrate 
the end-to-end data science workflow.

## Project overview

- **Goal:** Identify customers at risk of churning before they leave
- **Dataset:** IBM Telco Customer Churn — 7,043 customers, 21 features
- **Best model:** Logistic Regression (threshold 0.30) — ROC-AUC: 0.842
- **Key finding:** Contract type and tenure are the strongest predictors 
  of churn. Month-to-month customers churn at 42.7% vs 2.8% for 
  two-year contract customers.

## Results

| Model | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.842 | 0.75 | 0.53 | 0.62 |
| XGBoost | 0.819 | 0.79 | 0.48 | 0.59 |
| Random Forest | 0.814 | 0.71 | 0.52 | 0.60 |

The simplest model won. Logistic Regression outperformed both Random 
Forest and XGBoost — suggesting churn in this dataset is driven by 
largely linear relationships.

## Key insights

- Month-to-month customers churn at 15x the rate of two-year contract customers
- Churn is concentrated in the first 12 months — early intervention is critical
- Electronic check customers churn at 45.3% vs 15-19% for other payment methods
- Fiber optic internet customers are the highest risk internet service segment
- Tenure is the single strongest predictor of churn (SHAP value: 0.85)

## Project structure
```
telco-churn/
├── data/               # Raw data — not tracked (see setup below)
├── notebooks/          # One notebook per DS phase
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation.ipynb
├── outputs/            # Saved charts and figures
├── requirements.txt    # Python dependencies
└── README.md
```

## Setup

### 1. Clone the repo
```
git clone https://github.com/Drew-Zeimetz/telco-churn.git
cd telco-churn
```

### 2. Create and activate virtual environment
```
python -m venv venv
source venv/Scripts/activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Get the data
Download from kaggle.com/datasets/blastchar/telco-customer-churn
and place the CSV in the data/ folder.

### 5. Run notebooks in order
Start with 01_data_understanding.ipynb and run each notebook 
sequentially — each one saves output that the next notebook loads.

## Data science process

| Phase | Notebook | Description |
|---|---|---|
| Data understanding | 01 | Shape, dtypes, missing values, target distribution |
| EDA | 02 | 9 visualizations exploring churn drivers |
| Preprocessing | 03 | Encoding, feature engineering, train/test split, scaling |
| Modeling | 04 | Logistic Regression, Random Forest, XGBoost comparison |
| Evaluation | 05 | Confusion matrix, ROC curves, SHAP analysis |


## Tools used
Python, Pandas, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn, 
Jupyter, Git, GitHub
