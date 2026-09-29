# Reproducible analysis script for the TIMSS Türkiye forecasting study.
# Analytical period: 1999–2023. The 2003 achievement values are interpolated.

import numpy as np, pandas as pd
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

df=pd.read_csv("../data/timss_turkiye_mathematics.csv")
target="Mathematics_Score"
specs={'M0': [], 'M1': ['Value'], 'M2': ['Value', 'Like'], 'M3': ['Value', 'Like', 'Confident'], 'M4': ['Value', 'Like', 'Confident', 'Class_Size']}
test_years=[2011,2015,2019,2023]

def smape(a,p):
    a,p=np.asarray(a,float),np.asarray(p,float)
    return 100*np.mean(np.abs(a-p)/((np.abs(a)+np.abs(p))/2))
def dtw(a,b):
    D=np.full((len(a)+1,len(b)+1),np.inf); D[0,0]=0
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            D[i,j]=abs(a[i-1]-b[j-1])+min(D[i-1,j],D[i,j-1],D[i-1,j-1])
    return D[-1,-1]

rows=[]
for name,preds in specs.items():
    for year in test_years:
        tr=df[df.Year<year].copy(); te=df[df.Year==year].copy()
        train=pd.DataFrame({"ds":pd.to_datetime(tr.Year.astype(str)+"-01-01"),"y":tr[target].values})
        future=pd.DataFrame({"ds":pd.to_datetime(te.Year.astype(str)+"-01-01")})
        model=Prophet(growth="linear",yearly_seasonality=False,weekly_seasonality=False,
                      daily_seasonality=False,n_changepoints=0,uncertainty_samples=0)
        if preds:
            sc=StandardScaler(); Xtr=sc.fit_transform(tr[preds]); Xte=sc.transform(te[preds])
            for i,_ in enumerate(preds):
                rn=f"reg_{i+1}"; model.add_regressor(rn); train[rn]=Xtr[:,i]; future[rn]=Xte[:,i]
        model.fit(train)
        fc=float(model.predict(future)["yhat"].iloc[0])
        rows.append([name,year,float(te[target].iloc[0]),fc])

res=pd.DataFrame(rows,columns=["Model","Test_Year","Actual","Forecast"])
print(res)
for name in specs:
    s=res[res.Model==name]; a,p=s.Actual.values,s.Forecast.values; mse=mean_squared_error(a,p)
    print(name,{"MAE":mean_absolute_error(a,p),"MSE":mse,"RMSE":mse**0.5,"sMAPE":smape(a,p),"DTW":dtw(a,p)})
