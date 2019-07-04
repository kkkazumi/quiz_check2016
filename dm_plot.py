#python3 dm_plot.py
import numpy as np
from matplotlib import pyplot
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.patches import FancyArrowPatch

from conv_num import * #for id number


username = ['inusan', 'kumasan', 'kubosan','nekosan', 'sarada','test119', 'test120', 'test121', 'tomato', 'torisan', 'usagisan']

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = 2.0*(x-min)/(max-min) -1.0

    return result

def arrow(name, dm, ans):

    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    #ax2 = ax.twinx()

    mdiff = np.roll(ans,-1)-ans

    dm_nor = dm/(abs(max(dm)-min(dm)))

    mplus = ans + dm_nor
    half = (ans + mplus)/2.0
    corr=np.corrcoef(mdiff[:9],dm[:9])

    #print('dm',dm_nor)
    print('ans',ans)
    #print('mplus',mplus)
    print('mdiff',mdiff)
    print('dm',dm)

    print('corr',corr[1,0])

    ax.plot(range(1,10),half[0:9],'o',color="white",ms=1)
    ax.plot(range(0,10),ans,color="red",linestyle="dashed",label="answered M")
    ax.scatter(range(0,10),ans,color="red")#,legend="answered M")


    for x in range(9):
        ax.annotate('', xy=(x+0.5,half[x]),xytext=(x,ans[x]),
        #ax.annotate('', xy=(x+1,mplus[x+1]),xytext=(x,ans[x]),
            arrowprops=dict(shrink=0, width=1, headwidth=8, 
            headlength=10, connectionstyle='arc3',
            facecolor='black', edgecolor='black',label="marker")
        )

    ax.scatter([], [], c='black', marker=u'$\u2013\!\u25ba$', s=150, label='answered M')

    #ax.set_ylim(min(half),max(ans))

    #q=pyplot.quiver(range(0,10),ans,range(1,10),half)
    #q=pyplot.quiver([0,1],[-5,-10],[1,5],[1,5],scale=0.1)#,half)
    #q=pyplot.quiver(np.arange(0,10),ans,0.5,mplus-ans)#,half)
    #ax.legend(handles = [annotate], 
    #          handler_map={type(annotate) : AnnotationHandler(5)})
    #p = pyplot.quiverkey(q,1,16.5,50,"50 m/s",coordinatesred='data',color='r')
    ax.legend()
    ax.set_xlabel("time transition")
    ax.set_ylabel("intensity of mood")

    pyplot.title(name+'_'+str(round(corr[1,0],3)))
    print(conv_num(name))
    pyplot.show()

if __name__ == "__main__":
    name_list = conv_list(username)
    predicted_dm = np.loadtxt("./data/phi_dM.csv",delimiter=",",skiprows=1)
    for name in username:
        m_ans = np.loadtxt("./jiken/"+name+"/kibun_after.csv",delimiter=",")
        index_num = username.index(name)
        dm = predicted_dm[:,index_num]
        arrow(name,dm,m_ans*10)
