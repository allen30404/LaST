import sys
import csv
import pandas as pd


folder = "EXP_Record/Electricity_S_enmutiTrue_efuTrue_MVMmlp_demutiTrue_dfuFalse"
str1 = "LaST_ETTh1_ftS_sl201_ll0_pl720_lr0.001_bs32_ls128_dp0.2_enmutiTrue_demutiTrue_MVMmlp_enfuTrue_seed"
result=[]
with open('exp_record.txt','r') as f:
    for line in f:
        result.append(line.strip('\''))


with open(folder+'/'+str1+'.csv', 'w', newline='') as csvfile:
  # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)
    writer.writerow(['seed', 'mse', 'mae'])
    for i in range(len(result)):
        #result[i].find(str1)
        if(result[i].find(str1)!=-1):
            print(result[i]+'\n')
            writer.writerow([result[i].split('_')[-2].split('seed')[1], result[i+1].split(',')[0].split(':')[1], result[i+1].split(',')[1].split(':')[1]])


#读取文件数据
df = pd.read_csv(folder+'/'+str1+'.csv')
#按照列值排序
data=df.sort_values(by="mse" , ascending=False)
#把新的数据写入文件
data.to_csv(folder+'/'+str1+'.csv', mode='w', index=False)

#writer.writerow(['姓名', '身高', '體重'])
    