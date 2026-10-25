"""Week 3 · 03 solutions. Look only after a real attempt."""
import numpy as np
import pandas as pd


def summary(df):
    return df.shape[0], df.shape[1], int(df.isna().sum().sum())


def hot_night(df, t):
    return df[(df["shift"] == "night") & (df["temp"] > t)]


def parts_on_lines(df, lines):
    return df.loc[df["line"].isin(lines), "part"].tolist()


def top_k(df, col, k):
    return df.sort_values(col, ascending=False).head(k).reset_index(drop=True)


def add_columns(df):
    return df.assign(temp_f=df["temp"] * 9 / 5 + 32, hot=df["temp"] > 75)


def fill_temp_by_line(df):
    out = df.copy()
    out["temp"] = out["temp"].fillna(out.groupby("line")["temp"].transform("mean"))
    return out


def defect_rate_by(df, col):
    return df.groupby(col)["defect"].mean()


def line_stats(df):
    return df.groupby("line").agg(n=("part", "size"), mean_temp=("temp", "mean"), max_pressure=("pressure", "max"))


def attach_supervisor(df, sup):
    return df.merge(sup, on="line", how="left")


def defects_table(df):
    return df.pivot_table(index="line", columns="shift", values="defect", aggfunc="sum", fill_value=0)


def weekday_counts(df):
    return pd.to_datetime(df["date"]).dt.day_name().value_counts().sort_index()


def to_xy(df):
    d = pd.get_dummies(df[["temp", "pressure", "line", "shift", "defect"]].dropna(), columns=["line", "shift"], dtype=float)
    return d.drop(columns="defect").to_numpy(), d["defect"].to_numpy()
