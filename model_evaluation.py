import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, roc_curve, calibration_curve
import matplotlib.pyplot as plt

def evaluate(path_model, path_data):
    df = pd.read_csv(path_data)
    features = ['loan_amount','term_months','interest_rate','annual_income',
                'credit_score','employment_length_years','previous_defaults',
                'utilization','loan_to_income','income_per_month','high_utilization']
    X = df[features].fillna(0)
    y = df['default']
    d = joblib.load(path_model)
    # detect dict or model
    if isinstance(d, dict):
        scaler = d.get('scaler')
        model = d.get('lr')
        Xs = scaler.transform(X)
        probs = model.predict_proba(Xs)[:,1]
    else:
        model = d
        probs = model.predict_proba(X)[:,1]
    print('ROC-AUC', roc_auc_score(y, probs))

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    args = p.parse_args()
    evaluate(args.model, args.data)
