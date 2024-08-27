import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

def encode_labels_LabelEncoder(df: pd.DataFrame, cols: list):
    """Encode labels in columns of a dataframe with scikit LabelEncoder.

    Args:
        df (pd.DataFrame): given dataframe
        cols (list): names of the columns to encode

    Returns:
        df (pd.DataFrame): given dataframe with transformed columns
    """
    le = LabelEncoder()

    for col in cols:
        print(col)
        df[col] = le.fit_transform(df[col])

    return df

def convert_to_float(df: pd.DataFrame, cols: list):
    for col in cols:
        df.loc[df[col] == '-', col] = 0
        df[col] = df[col].astype('float')

    return df


def preprocess_df(df: pd.DataFrame, filter_100=False):
    if filter_100:
        df = df.loc[df['RR'] <= 100]

    cols = ['BOND_COUPON', 'OFFERING_AMT']
    df = convert_to_float(df=df, cols=cols)

    cols = ['CompanyName', 'csp_CIQ', 'Type']
    df = encode_labels_LabelEncoder(df=df, cols=cols)
   
    return df

def scale_data(df: pd.DataFrame):
    scaler = StandardScaler()
    arguments_list = df.columns.to_list()
    df_scaled = pd.DataFrame()
    df_scaled[arguments_list] = scaler.fit_transform(df[arguments_list])

    return df_scaled

def split_labels(df: pd.DataFrame):
    argument_filter =['RR', 'Ddate']
    arguments_list = [x for x in df.columns.to_list() if x not in argument_filter]
    labels_list = ['RR']

    X = pd.DataFrame()
    y = pd.DataFrame()

    X[arguments_list] = df[arguments_list]
    y = df[labels_list]

    return X, y


def train_test_splitting(X: pd.DataFrame, y: pd.DataFrame, scenario: str, cut_off_year: int = 2011, df=None):
    
    if scenario == 'out_of_sample':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    elif scenario == 'out_of_time':
        cut_off = max(df.loc[df['Ddate'].dt.year <= cut_off_year].index)
            
        X_train = X[:cut_off]
        X_test = X[cut_off:]
        y_train = y[:cut_off]
        y_test = y[cut_off:]
    
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    print(f'Samples for X_train: {len(X_train)}')
    print(f'Samples for X_test: {len(X_test)}')
    print(f'Samples for y_train: {len(y_train)}')
    print(f'Samples for y_test: {len(y_test)}')

    return X_train, X_test, y_train, y_test
