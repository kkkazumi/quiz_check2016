import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd

def get_spl(x_data,y_data):
  f_sci = interpolate.interp1d(x_data,y_data,kind="cubic")

USER_NUM = 9

FACTOR_NUM = 10
FACE_TYPE = 4

DIR_PATH = "../../../est_pred_pc/data/"
face_type_list=["happy","surprised","angry","sad"]
factor_type_list=["trial num","rate of win","rate of encourage behavior","rate of sympathetic behavior", "rate of teasing behavior","rate of un-related behavior","rate of no behavior","total point","consecutive wins","consecutive loses"]

#def ret_data(x_data,y_data,mental_data):
def ret_data(df):
  ##sorted x and y
  x_data=np.array(df["factor"])+0.0001
  y_data=np.array(df["signal"])
  mental_data=np.array(df["mental"])

  x_data_2=x_data[np.argsort(x_data)]
  y_data_2=y_data[np.argsort(x_data)]
  mental_2=mental_data[np.argsort(x_data)]

  res = np.polyfit(x_data_2,y_data_2,2)
  y_res=np.poly1d(res)(x_data_2)
  return x_data_2,y_res

def show_graph(username,factor_data,signal_data,mental_data,thr):
  for factor_type in range(FACTOR_NUM):
    for signal_type in range(FACE_TYPE):
      #for m in range(10):
      x_data = factor_data[:,factor_type]
      y_data = signal_data[:,signal_type]

      df=pd.DataFrame(x_data,columns=["factor"])
      df["signal"]=y_data
      df["mental"]=mental_data

      df_sml=df.query(str(thr[1])+'>mental>='+str(thr[0]))
      val_sml=int(df_sml["factor"].count())
      df_mid=df.query(str(thr[2])+'>mental>='+str(thr[1]))
      val_mid=int(df_mid["factor"].count())
      df_big=df.query(str(thr[3])+'>=mental>='+str(thr[2]))
      val_big=int(df_big["factor"].count())

      if(val_sml*val_mid*val_big>0):
        x_sml,y_res_sml=ret_data(df_sml)
        x_mid,y_res_mid=ret_data(df_mid)
        x_big,y_res_big=ret_data(df_big)

        plt.plot(x_sml,y_res_sml,color='blue',label="sml",linestyle="dotted")
        plt.plot(x_mid,y_res_mid,color='green',label="mid",linestyle="dashed")
        plt.plot(x_big,y_res_big,color='red',label="big",linestyle="solid")
      plt.scatter(x_data,y_data,c=mental_data)
      plt.legend()

      plt.ylabel("face data("+face_type_list[signal_type]+")")
      plt.xlabel("factor data("+factor_type_list[factor_type]+")")

      filename="./plot/graph_u"+str(username)+"_f"+str(factor_type)+"_s"+str(signal_type)+".png"
      plt.savefig(filename)
      print("u",username,"f",factor_type,"s",signal_type)
      plt.clf()


filename='mental_thr_memo.csv'
data=np.loadtxt(filename,delimiter=",")

for username in range(1,USER_NUM+1):
  thr=[0,0.4,0.7,1]
  thr[1]=data[username,1]
  thr[2]=data[username,2]
  print(thr)

  factor_file = DIR_PATH+str(username)+"/factor_before.csv"
  signal_file = DIR_PATH+str(username)+"/signal_before.csv"
  mental_file = DIR_PATH+str(username)+"/kibun_before.csv"

  factor_data = np.loadtxt(factor_file,delimiter="\t")
  signal_data = np.loadtxt(signal_file,delimiter="\t")
  mental_data = np.loadtxt(mental_file,delimiter="\t")
  df_mental = pd.DataFrame(mental_data,columns=["mental"])


  plt.hist(df_mental["mental"])
  #plt.vlines([df_mental.describe()["mental"]["mean"]], 0, 30, "blue", linestyles='dashed')
  plt.vlines(thr[1], 0, 30, "red", linestyles='dashed')
  plt.vlines(thr[2], 0, 30, "green", linestyles='dashed')
  plt.show()
