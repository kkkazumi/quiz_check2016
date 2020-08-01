import numpy as np
import matplotlib.pyplot as plt


def out_gosa(graph_type):

  if graph_type == 'all':
    phi_data = np.zeros(5)
    nn_data = np.zeros(5)

    for i in range(9):
      #plt.subplot(3,3,i+1)

      usernum = i+1

      for t_num in (25,26,27,28,29):
        test_num = t_num

        phi_file = "./jrm_test/"+str(usernum)+"/phi_sign-"+str(t_num)+".csv"
        nn_file = "./jrm_test/"+str(usernum)+"/nn_sign-"+str(t_num)+".csv"

        phi_sign = np.loadtxt(phi_file,delimiter=",")
        nn_sign = np.loadtxt(nn_file,delimiter=",")

        phi_data[t_num-25] += np.sum(phi_sign)#/29.0
        nn_data[t_num-25] += np.sum(nn_sign)#/29.0

      #plt.bar(x,phi_sign,width=0.5,label="phi")
      #plt.bar(x+0.5,nn_sign,width=0.5,label="nn")
      plt.xlabel("number of su data(id:"+str(i+1)+")")
      plt.ylabel("average of correct sign")

    x = np.linspace(0,5,5)+25
    plt.bar(x-0.25,nn_data/270.0,label='nn',width = 0.5)
    plt.bar(x+0.25,phi_data/270.0,label='phi',width=0.5)

#plt.bar(x,phi_sign-nn_sign)

#plt.title(str(usernum)+"-"+str(test_num))
    plt.legend()
    #plt.show()
    plt.savefig('sign_check25-29.eps')

  elif graph_type == 'each':
    phi_data = np.zeros((9,5))
    nn_data = np.zeros((9,5))


    for t_num in (25,26,27,28,29):
        test_num = t_num

        for i in range(9):
            usernum = i+1

            phi_file = "./jrm_test/"+str(usernum)+"/phi_sign-"+str(t_num)+".csv"
            nn_file = "./jrm_test/"+str(usernum)+"/nn_sign-"+str(t_num)+".csv"

            phi_sign = np.loadtxt(phi_file,delimiter=",")
            nn_sign = np.loadtxt(nn_file,delimiter=",")

            phi_data[i,t_num-25] = np.sum(phi_sign)/29.0
            nn_data[i,t_num-25] = np.sum(nn_sign)/29.0

            #plt.bar(x,phi_sign,width=0.5,label="phi")
            #plt.bar(x+0.5,nn_sign,width=0.5,label="nn")
            plt.xlabel("user id")
            plt.ylabel("average of correct sign")

        print(t_num)
        x = np.linspace(1,9,9)
        plt.bar(x-0.2,nn_data[:,t_num-25],label='nn',width = 0.4)
        plt.bar(x+0.2,phi_data[:,t_num-25],label='phi',width=0.4)

        plt.legend()
        #plt.show()
        plt.savefig('sign_check-'+str(t_num)+'.eps')
        plt.clf()


if __name__ == '__main__':
  out_gosa('all')
