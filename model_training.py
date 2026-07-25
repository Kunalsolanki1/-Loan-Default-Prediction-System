import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from feature_engineering import add_features

def train_models(path):
    df = pd.read_csv(path)
    df = add_features(df)
    features = ['loan_amount','term_months','interest_rate','annual_income',
                'credit_score','employment_length_years','previous_defaults',
                'utilization','loan_to_income','income_per_month','high_utilization']
    X = df[features].fillna(0)
    y = df['default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_s, y_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Save models
    joblib.dump({'scaler': scaler, 'lr': lr}, 'outputs/logistic_model.joblib')
    joblib.dump(rf, 'outputs/rf_model.joblib')

    # Evaluate
    lr_probs = lr.predict_proba(X_test_s)[:,1]
    rf_probs = rf.predict_proba(X_test)[:,1]
    print('LR ROC-AUC:', roc_auc_score(y_test, lr_probs))
    print('RF ROC-AUC:', roc_auc_score(y_test, rf_probs))
    print('\nLogistic Classification Report:')
    print(classification_report(y_test, lr.predict(X_test_s)))
    print('\nRandom Forest Classification Report:')
    print(classification_report(y_test, rf.predict(X_test)))
    return

if __name__ == '__main__':
    import argparse, os
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    args = p.parse_args()
    os.makedirs('outputs', exist_ok=True)
    train_models(args.data)
