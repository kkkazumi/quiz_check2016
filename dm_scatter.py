#python3 dm_plot.py
import numpy as np
from matplotlib import pyplot

username = ['inusan', 'kumasan', 'nekosan', 'test119', 'test120', 'test121', 'tomato', 'torisan', 'usagisan']

def arrow(name, m, dm, ans):

    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    #ax2 = ax.twinx()

    mdiff = np.roll(ans,-1)-ans

    ax.scatter(mdiff,dm,color='red')
    ax.set_xlabel('andwered')
    ax.set_ylabel('predicted')
    corr=np.corrcoef(mdiff,dm)
    print('corr',corr)
    pyplot.title(name)
    pyplot.show()

if __name__ == "__main__":
    predicted_dm = np.loadtxt("./data/phi_dM.csv",delimiter=",",skiprows=1)
    print(predicted_dm[:,0])
    for name in username:
        m_ans = np.loadtxt("./jiken/"+name+"/kibun_after.csv",delimiter=",")
        m = np.loadtxt("./jiken/"+name+"/m_pred.csv",delimiter=",")
        phi_M = np.loadtxt("./jiken/"+name+"/m_pred.csv",delimiter=",")
        index_num = username.index(name)
        dm = predicted_dm[:,index_num]
        arrow(name,m,dm,m_ans)
