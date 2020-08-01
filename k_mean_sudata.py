import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


user_name = 0
set_num = 29

set_num_list = (25,26,27,28,29)

#factor = np.zeros((25*9,250))
factor_vec = np.zeros((9*set_num*len(set_num_list),len(set_num_list)*10))
f_var = np.zeros(9*set_num*len(set_num_list))
s_var = np.zeros(9*set_num*len(set_num_list))
est_q = np.zeros(9*set_num*len(set_num_list))

num = np.zeros(9*25*5)

for set_num in set_num_list:
  _s = set_num
  num[int(_s*25*9):int(_s*25*9)+25*9]=set_num
  #num[int(_s*25*9):int(_s*25*9)+25*9]=set_num
  for user_name in range(9):
    #standard for
    filepath_t = "./jrm_test/"+str(user_name+1)+"/phi_point-" +str(set_num) +".csv"
    est_q[int(_s*25*9+user_name*25):int(_s*25*9+user_name*25)+25] = np.loadtxt(filepath_t,delimiter=",")

    for test_num in range(25):

      filepath_ftes = "./jrm_test/"+str(user_name+1)+"/factor_test" + str(set_num) + "-" +str(test_num) +".csv"
      filepath_stes = "./jrm_test/"+str(user_name+1)+"/face_test" + str(set_num) + "-" +str(test_num) +".csv"

      filepath_f = "./jrm_test/"+str(user_name+1)+"/factor_train" + str(set_num) + "-" +str(test_num) +".csv"
      filepath_s = "./jrm_test/"+str(user_name+1)+"/face_train" + str(set_num) + "-" +str(test_num) +".csv"
      t=int(_s*25*9+user_name*25+test_num)

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
