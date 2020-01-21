import numpy as np
import matplotlib.pyplot as plt

#ok_list = np.array([[0,11],[0,12],[2,18],[4,7],[4,11],[4,24],[5,11],[5,20],[6,3],[7,1],[7,19]])
ok_list = np.array([[7,1]])
#ok_list = np.array([[0,11]])

for user_name in range(11):
  dir_name = "./jrm_test/" + str(user_name+1)
  #for set_num in (5,10,15,20,25,30):
  set_num = 40
  for i in range(30):


    #i = ok_list[user_name,1]

    i_csv = str(set_num) + "-" + str(i) + ".csv"

    mood_test= np.loadtxt(dir_name + "/mood_test" + i_csv ,delimiter=',')
    #mood_est = np.loadtxt(dir_name + "/estimated_dummy" + i_csv, delimiter=',')
    mood_est = np.loadtxt(dir_name + "/estimated_phi" + i_csv, delimiter=',')

    #print pearsonr(mood_test,mood_est)

    fz=12
    label=np.linspace(18,27,10)

    fig,ax = plt.subplots()
    plt.plot(label,mood_test,color='limegreen',marker='o',label='self-asessed mood')
    plt.plot(label,mood_est/10.0,color='green',linestyle='dashed',marker='o',markerfacecolor='white',label='estimated mood')
    plt.ylabel('estimated/self-assessed mood',fontsize=fz)
    plt.ylim(-0.02,1.2)
    plt.legend()
    """
    g1=ax.plot(label,mood_test,color='limegreen',marker='o',label='self-asessed mood')
    ax2 = ax.twinx()
    g2=ax2.plot(label,mood_est/10.0,color='green',linestyle='dashed',marker='o',markerfacecolor='white',label='estimated mood')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='best',fontsize=fz)
    #ax2.set_ylabel('self-asessed mood',fontsize=fz)
    ax.set_ylabel('estimated mood',fontsize=fz)
    ax.set_ylim(-0.02,1.2)
    ax2.set_ylim(-0.02,1.2)
    ax.set_xlabel('question number',fontsize=fz)
    plt.tick_params(labelsize=fz)
    """
    plt.show()
    #plt.savefig('2nd_hi2.eps')
    
