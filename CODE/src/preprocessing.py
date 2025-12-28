import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def apply_log_return(df, column="Close"):
    df["log_return"] = np.log(df[column]).diff()
    return df.dropna()

def scale_features(df, column="log_return"):
    scaler = StandardScaler()
    df["scaled"] = scaler.fit_transform(df[[column]])
    return df