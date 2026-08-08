"""
features.py - Feature engineering helper functions
Author: Dhruv Mehta
"""
import pandas as pd
import numpy as np

def add_rolling_features(df, cols, windows=[3,5], group_col="element", sort_col="GW"):
    df = df.sort_values([group_col, sort_col])
    for col in cols:
        for w in windows:
            shifted = df.groupby(group_col)[col].shift(1)
            df[f"{col}_roll{w}"] = shifted.groupby(df[group_col]).transform(
                lambda x: x.rolling(w, min_periods=1).mean())
    return df

def add_availability_flag(df, minutes_col="minutes", group_col="element"):
    df["minutes_lag1"] = df.groupby(group_col)[minutes_col].shift(1)
    df["minutes_lag2"] = df.groupby(group_col)[minutes_col].shift(2)
    df["availability_flag"] = ((df["minutes_lag1"] > 0) | (df["minutes_lag2"] > 0)).astype(int)
    return df

def add_per90_features(df, minutes_col="minutes"):
    mins = df[minutes_col].clip(lower=1)
    df["goals_per_90"]   = df["goals_scored"] / (mins/90)
    df["assists_per_90"] = df["assists"] / (mins/90)
    df["saves_per_90"]   = df["saves"] / (mins/90)
    df["bonus_per_90"]   = df["bonus"] / (mins/90)
    return df

def encode_positions(df, pos_col="position"):
    pos_dummies = pd.get_dummies(df[pos_col], prefix="pos")
    df = pd.concat([df, pos_dummies], axis=1)
    for col in ["pos_GKP","pos_DEF","pos_MID","pos_FWD"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
        else:
            df[col] = 0
    return df
