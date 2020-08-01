import numpy as np
import matplotlib.pyplot as plt


INPUT_DIM = 11
MID_UNIT = 15
OUTPUT_DIM = 1
EPOCH = 50000

def m_conf(mental):
  return mental/40.0+0.5

def conf_m(pred_mental):
  return (pred_mental-0.5)

for name_num in (0,1,2,3,4,5,6,7,8):
  dir_name = "./jrm_test/" + str(name_num+1)
  #for set_num in (5,10,15,20,25,30,35,40):
  #for set_num in (5,10,15,20,25):

  set_num = 25
  r = np.zeros(25)
  p = np.zeros(25)
  for i in range(25):#calculate the avelage value of estimated accuracy for datum in this loop
    i_csv = str(set_num) + "-" + str(i) + ".csv"

    #test to output
    _mood_test = np.loadtxt(dir_name + "/TESTestimated_phi" + i_csv ,delimiter=',')
    m_ans = np.diff(m_conf(_mood_test))

    mood_pred= conf_m(np.loadtxt(dir_name + "/estimated_dM" + i_csv ,delimiter=','))
    x=np.linspace(0,3,4)

    plt.bar(x,m_ans,width=0.4,label="answer")
    plt.bar(x+0.5,mood_pred,width=0.4,label="predicted")
    plt.legend()
    plt.show()
