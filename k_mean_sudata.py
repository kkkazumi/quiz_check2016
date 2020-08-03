#python3
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

users = range(9)
set_num = 29
set_num_list = (25,26,27,28,29)
factor_num = 10

#factor = np.zeros((25*9,250))
factor_vec = np.zeros((len(users)*set_num*len(set_num_list),factor_num))
#factor_vec = np.zeros((len(users)*set_num*len(set_num_list),len(set_num_list)*factor_num))
f_var = np.zeros(len(users)*set_num*len(set_num_list))
s_var = np.zeros(len(users)*set_num*len(set_num_list))
est_q = np.zeros(len(users)*set_num*len(set_num_list))

num = np.zeros(len(users)*set_num*len(set_num_list))

for st in set_num_list:
  _s = st-25
  num[int(_s*set_num*len(users)):int(_s*set_num*len(users))+set_num*len(users)]=st
  #num[int(_s*25*9):int(_s*25*9)+25*9]=set_num
  for user_name in users:
    #standard for
    filepath_t = "./jrm_test/"+str(user_name+1)+"/phi_point-" +str(st) +".csv"
    in_i = int(_s*set_num*len(users)+user_name*set_num)
    out_i = int(_s*set_num*len(users)+user_name*set_num)+set_num
    est_q[int(_s*set_num*len(users)+user_name*set_num):int(_s*set_num*len(users)+user_name*set_num)+set_num] = np.loadtxt(filepath_t,delimiter=",")


    for test_num in range(set_num):

      filepath_ftes = "./jrm_test/"+str(user_name+1)+"/factor_test" + str(st) + "-" +str(test_num) +".csv"
      filepath_stes = "./jrm_test/"+str(user_name+1)+"/face_test" + str(st) + "-" +str(test_num) +".csv"

      filepath_f = "./jrm_test/"+str(user_name+1)+"/factor_train" + str(st) + "-" +str(test_num) +".csv"
      filepath_s = "./jrm_test/"+str(user_name+1)+"/face_train" + str(st) + "-" +str(test_num) +".csv"
      t=int(_s*set_num*len(users)+user_name*set_num+test_num)

      _factor = np.loadtxt(filepath_f,delimiter=",")
      _factor_tes = np.loadtxt(filepath_ftes,delimiter=",")
      _signal = np.loadtxt(filepath_s,delimiter=",")
      f_var[t] = np.var(_factor)
      s_var[t] = np.var(_signal)

      factor_vec[t,:]=_factor_tes.reshape(-1)
      #factor_vec[user_name*25+test_num,:,:]=_factor

pca = PCA(n_components=2).fit(factor_vec)
values = pca.transform(factor_vec)
print(values.shape)

#pred = KMeans(n_clusters=10).fit_predict(factor)
#clus = pred.reshape(9,-1)

d = {'est_qual':est_q,'fact_var':f_var,'sig_var':s_var,'numb':num,'su_1':values[:,0],'su_2':values[:,1]}
df= pd.DataFrame(data=d)
print(df.corr())

from factor_analyzer import FactorAnalyzer

columns = df.columns
n=3
fa = FactorAnalyzer(n_factors=n,rotation="promax",impute="drop")
fa.fit(df.values)
print(fa.loadings_)

#import seaborn as sns
#cm = sns.light_palette('red',as_cmap=True)
#df.style.background_gradient(cmap=cm) 
