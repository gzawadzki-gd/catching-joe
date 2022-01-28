import pandas as pd
import numpy as np
from src.constraints import fav_sites


def unwrap_location(df):
    df[["country", "city"]] = df["location"].str.split("/", expand=True)
    return df


def fix_locale(df):
    df["locale"] = df["locale"].str.replace("_", "-").replace(" ", "")
    return df


def unwrap_date(df):
    df[["dayofweek", "day", "month"]] = df["date"].apply(
        lambda x: pd.Series([x.dayofweek, x.day, x.month])
    )
    return df


def time_to_decimal(df):
    df["hour"] = df["time"].apply(
        lambda x: pd.to_datetime(x).hour + pd.to_datetime(x).minute / 60
    )
    return df


def clean_sites(x):
    if x in fav_sites:
        return x
    elif x == "":
        return "empty"
    else:
        return "other"


def get_num_sites(history):
    return history.notnull().sum(axis=1)


def get_session_len(history):
    return history.apply(lambda x: x.str["length"]).sum(axis=1)


def unwrap_sites(df):
    history = df["sites"].apply(pd.Series)
    df["sites_num"] = get_num_sites(history)
    df["session_len"] = get_session_len(history)
    for i in history.columns:
        df["site_" + str(i)] = (
            history.fillna("").apply(lambda x: x.str["site"])[i].apply(clean_sites)
        )
    return df


def add_is_joe(df):
    df["is_joe"] = df["user_id"].apply(lambda x: 1 if x == 0 else 0)
    return df


def preprocess(df):
    df = unwrap_location(df)
    df = fix_locale(df)
    df = time_to_decimal(df)
    df = unwrap_date(df)
    df = unwrap_sites(df)
    df = add_is_joe(df)
    return df
