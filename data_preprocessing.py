import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def basic_cleaning(df):
    # Example cleaning: fill na, correct dtypes
    df = df.copy()
    num_cols = df.select_dtypes(include=['float64','int64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    args = p.parse_args()
    df = load_data(args.data)
    df = basic_cleaning(df)
    print('Loaded', df.shape)
