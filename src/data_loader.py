import pandas as pd

def load_data(df):
    df = pd.read_csv('D:\PYTHON DATA\Recomendador\dataset\cannabis_clean.csv')
    return df
