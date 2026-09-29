# Reproducible analysis script for the TIMSS Türkiye forecasting study.
# Analytical period: 1999–2023. The 2003 achievement values are interpolated.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

def exploratory_ordering(csv_path, target, predictors):
    df = pd.read_csv(csv_path)
    X = StandardScaler().fit_transform(df[predictors])
    y = StandardScaler().fit_transform(df[[target]]).ravel()
    fit = SARIMAX(y, exog=X, order=(0,0,0), trend="n",
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    out = pd.DataFrame({"Predictor": predictors, "Beta": fit.params[:len(predictors)]})
    out["Abs_Beta"] = out["Beta"].abs()
    return out.sort_values("Abs_Beta", ascending=False)

print("SCIENCE")
print(exploratory_ordering(
    "../data/timss_turkiye_science.csv",
    "Science_Score",
    ["Class_Size","Like","Value","Confident"]
))
print("\nMATHEMATICS")
print(exploratory_ordering(
    "../data/timss_turkiye_mathematics.csv",
    "Mathematics_Score",
    ["Value","Like","Confident","Class_Size"]
))
