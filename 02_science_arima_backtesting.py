# Reproducible analysis script for the TIMSS Türkiye forecasting study.
# Analytical period: 1999–2023. The 2003 achievement values are interpolated.

import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv("../data/timss_turkiye_science.csv")
target = "Science_Score"
specs = {'M0': [], 'M1': ['Class_Size'], 'M2': ['Class_Size', 'Like'], 'M3': ['Class_Size', 'Like', 'Value'], 'M4': ['Class_Size', 'Like', 'Value', 'Confident']}
test_years = [2011,2015,2019,2023]

def smape(a,p):
    a,p=np.asarray(a,float),np.asarray(p,float)
    return 100*np.mean(np.abs(a-p)/((np.abs(a)+np.abs(p))/2))

def dtw(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    D=np.full((len(a)+1,len(b)+1),np.inf); D[0,0]=0
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            D[i,j]=abs(a[i-1]-b[j-1])+min(D[i-1,j],D[i,j-1],D[i-1,j-1])
    return D[-1,-1]

rows=[]
for name,preds in specs.items():
    for year in test_years:
        tr=df[df.Year<year].copy(); te=df[df.Year==year].copy()
        y=tr[target].values
        if preds:
            sc=StandardScaler(); Xtr=sc.fit_transform(tr[preds]); Xte=sc.transform(te[preds])
            fit=SARIMAX(y, exog=Xtr, order=(0,1,0), trend="n",
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False,maxiter=2000)
            if not fit.mle_retvals.get("converged",True):
                fit=SARIMAX(y, exog=Xtr, order=(0,1,0), trend="n",
                            enforce_stationarity=False, enforce_invertibility=False).fit(method="powell",disp=False,maxiter=5000)
            fc=float(fit.forecast(1, exog=Xte)[0])
        else:
            fit=SARIMAX(y, order=(0,1,0), trend="n",
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False,maxiter=2000)
            fc=float(fit.forecast(1)[0])
        rows.append([name,year,float(te[target].iloc[0]),fc])

res=pd.DataFrame(rows,columns=["Model","Test_Year","Actual","Forecast"])
print(res)
for name in specs:
    s=res[res.Model==name]
    a,p=s.Actual.values,s.Forecast.values
    mse=mean_squared_error(a,p)
    print(name, {
        "MAE":mean_absolute_error(a,p),
        "MSE":mse,
        "RMSE":mse**0.5,
        "sMAPE":smape(a,p),
        "DTW":dtw(a,p)
    })
