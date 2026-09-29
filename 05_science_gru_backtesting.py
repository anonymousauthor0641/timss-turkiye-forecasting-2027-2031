# Reproducible analysis script for the TIMSS Türkiye forecasting study.
# Analytical period: 1999–2023. The 2003 achievement values are interpolated.

import random, numpy as np, pandas as pd, tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping

SEEDS=[5,11,17,23,29]; LOOKBACK=2; UNITS=8; EPOCHS=120; PATIENCE=10; LR=0.001
df=pd.read_csv("../data/timss_turkiye_science.csv"); target="Science_Score"; specs={'M0': [], 'M1': ['Class_Size'], 'M2': ['Class_Size', 'Like'], 'M3': ['Class_Size', 'Like', 'Value'], 'M4': ['Class_Size', 'Like', 'Value', 'Confident']}; test_years=[2011,2015,2019,2023]

def seq(X,y):
    xs,ys=[],[]
    for i in range(LOOKBACK,len(X)): xs.append(X[i-LOOKBACK:i]); ys.append(y[i])
    return np.asarray(xs),np.asarray(ys)

def one(train,preds,seed):
    tf.keras.backend.clear_session()
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
    cols=[target]+preds
    sx,sy=StandardScaler(),StandardScaler()
    X=sx.fit_transform(train[cols]); y=sy.fit_transform(train[[target]]).ravel()
    Xs,ys=seq(X,y)
    model=Sequential([Input(shape=(LOOKBACK,len(cols))),GRU(UNITS,activation="tanh"),Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),loss="mse")
    model.fit(Xs,ys,epochs=EPOCHS,batch_size=1,shuffle=False,verbose=0,
              callbacks=[EarlyStopping(monitor="loss",patience=PATIENCE,min_delta=1e-5,restore_best_weights=True)])
    pred=float(model(np.expand_dims(X[-LOOKBACK:],0),training=False).numpy().ravel()[0])
    return float(sy.inverse_transform([[pred]])[0,0])

rows=[]
for name,preds in specs.items():
    for year in test_years:
        tr=df[df.Year<year].copy(); actual=float(df.loc[df.Year==year,target].iloc[0])
        for seed in SEEDS:
            rows.append([name,year,seed,actual,one(tr,preds,seed)])

res=pd.DataFrame(rows,columns=["Model","Test_Year","Seed","Actual","Forecast"])
summary=res.groupby(["Model","Test_Year"]).agg(Actual=("Actual","first"),Forecast_Mean=("Forecast","mean"),Forecast_SD=("Forecast","std")).reset_index()
print(summary)
