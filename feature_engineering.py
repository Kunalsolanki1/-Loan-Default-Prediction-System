import pandas as pd
import numpy as np

def add_features(df):
    df = df.copy()
    # Risk ratios and normalized features
    df['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1)
    df['income_per_month'] = df['annual_income'] / 12
    df['credit_score_band'] = pd.cut(df['credit_score'], bins=[299,549,649,699,749,900],
                                     labels=['very_poor','poor','fair','good','excellent'])
    # rolling-like features (example, for synthetic data we simulate)
    df['high_utilization'] = (df['utilization'] > 0.6).astype(int)
    return df

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    args = p.parse_args()
    df = pd.read_csv(args.data)
    df = add_features(df)
    print('Features added', df.columns.tolist())
