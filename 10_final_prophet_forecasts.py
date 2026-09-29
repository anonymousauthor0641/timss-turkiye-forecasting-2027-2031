# Reproducible analysis script for the TIMSS Türkiye forecasting study.
# Analytical period: 1999–2023. The 2003 achievement values are interpolated.

import pandas as pd
from prophet import Prophet

def fit_forecast(csv_path, target):
    df=pd.read_csv(csv_path)
    train=pd.DataFrame({
        "ds":pd.to_datetime(df["Year"].astype(str)+"-01-01"),
        "y":df[target].values
    })
    model=Prophet(growth="linear",yearly_seasonality=False,weekly_seasonality=False,
                  daily_seasonality=False,n_changepoints=0,uncertainty_samples=0)
    model.fit(train)
    future=pd.DataFrame({"ds":pd.to_datetime(["2027-01-01","2031-01-01"])})
    return model.predict(future)[["ds","yhat"]]

science=fit_forecast("../data/timss_turkiye_science.csv","Science_Score")
math=fit_forecast("../data/timss_turkiye_mathematics.csv","Mathematics_Score")
out=pd.DataFrame({
    "TIMSS_Cycle":[2027,2031],
    "Science_Score":science["yhat"].values,
    "Mathematics_Score":math["yhat"].values
})
print(out.round(2))
out.to_csv("../results/final_forecasts.csv",index=False)
