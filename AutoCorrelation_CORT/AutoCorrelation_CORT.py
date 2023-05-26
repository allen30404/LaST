import pandas as pd
import matplotlib.pyplot as plt
import numpy as np




 

df = pd.read_csv('../datasets/ETTh1.csv', sep=',', usecols=['OT'])
fig = plt.figure() #定義一個圖像窗口

X = df[0:120].to_numpy()
X_1 = df[1:121].to_numpy()
X_T = df[8763:8883].to_numpy()
X_T_1 = df[8764:8884].to_numpy()
Delta_X = X_1-X
Delta_X_T = X_T_1-X_T
plt.plot(np.linspace(0,120,120),X,label="X") #X
plt.plot(np.linspace(0,120,120),X_T,label="X_T") #X_T
fig.savefig('X_and_X_T.png')
plt.clf()
plt.plot(np.linspace(0,120,120),Delta_X,label="Delta_X") #Delta_X
plt.plot(np.linspace(0,120,120),Delta_X_T,label="Delta_X_T") #Delta_X_T
fig.savefig('Delta_X_and_Delta_X_T.png')
plt.clf()

