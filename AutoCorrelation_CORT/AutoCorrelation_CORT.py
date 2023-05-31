import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math



 

df = pd.read_csv('../datasets/ETTh1.csv', sep=',', usecols=['OT'])
fig = plt.figure(figsize=(25,5)) #定義一個圖像窗口

X = df[360:720].to_numpy()
X_1 = df[361:721].to_numpy()
X_T = df[9123:9483].to_numpy()
X_T_1 = df[9124:9484].to_numpy()
Delta_X = X_1-X
Delta_X_T = X_T_1-X_T
plt.plot(np.linspace(0,360,360),X,label="X1") #X
plt.plot(np.linspace(0,360,360),X_T,label="X2") #X_T
plt.legend()
fig.savefig('X1_and_X2.png')
plt.clf()
plt.plot(np.linspace(0,360,360),Delta_X,label="Delta_X1") #Delta_X
plt.plot(np.linspace(0,360,360),Delta_X_T,label="Delta_X2") #Delta_X_T
plt.legend()
fig.savefig('Delta_X1_and_Delta_X2.png')
plt.clf()

CORT_Delta_X = math.sqrt(sum(Delta_X*Delta_X))
CORT_Delta_X_T = math.sqrt(sum(Delta_X_T*Delta_X_T))
CORT_Delta_X_and_X_T = sum(Delta_X*Delta_X_T)
#print(CORT_Delta_X)
#print(CORT_Delta_X_T)
#print(CORT_Delta_X_and_X_T)
#print(CORT_Delta_X_and_X_T/(CORT_Delta_X*CORT_Delta_X_T))

X = df[360:720].to_numpy()
X_12lag = df[372:732].to_numpy()
X_S = df[9123:9483].to_numpy()
X_S_12lag = df[9135:9495].to_numpy()

X_sub_X_mean = X-np.mean(X)
X_12lag_sub_X_mean = X_12lag-np.mean(X)
X_S_sub_X_mean = X_S-np.mean(X_S)
X_S_12lag_sub_X_mean = X_S_12lag-np.mean(X_S)

plt.plot(np.linspace(0,360,360),X_sub_X_mean*X_12lag_sub_X_mean,label="Axx(x1)") #X
plt.plot(np.linspace(0,360,360),X_S_sub_X_mean*X_S_12lag_sub_X_mean,label="Axx(x2)") #X_T
plt.legend()
fig.savefig('Axx(x1)_and_Axx(X2).png')