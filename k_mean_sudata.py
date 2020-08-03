#python3
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

users = range(9)
set_num = 29
set_num_list = (25,26,27,28,29)
factor_num = 10
face_num = 4

#factor = np.zeros((25*9,250))
factor_vec = np.zeros((len(users)*set_num*len(set_num_list),factor_num))
#factor_vec = np.zeros((len(users)*set_num*len(set_num_list),len(set_num_list)*factor_num))
f_var = np.zeros((factor_num,len(users)*set_num*len(set_num_list)))
s_var = np.zeros((face_num,len(users)*set_num*len(set_num_list)))

f_test = np.zeros((factor_num,len(users)*set_num*len(set_num_list)))

est_q = np.zeros(len(users)*set_num*len(set_num_list))
num = np.zeros(len(users)*set_num*len(set_num_list))

for st in set_num_list:
  _s = st-25
  num[int(_s*set_num*len(users)):int(_s*set_num*len(users))+set_num*len(users)]=st
  for user_name in users:
    #standard for
    filepath_t = "./jrm_test/"+str(user_name+1)+"/phi_point-" +str(st) +".csv"
    in_i = int(_s*set_num*len(users)+user_name*set_num)
    out_i = int(_s*set_num*len(users)+user_name*set_num)+set_num
    est_q[int(_s*set_num*len(users)+user_name*set_num):int(_s*set_num*len(users)+user_name*set_num)+set_num] = 100-np.loadtxt(filepath_t,delimiter=",")


    for test_num in range(set_num):

      filepath_ftes = "./jrm_test/"+str(user_name+1)+"/factor_test" + str(st) + "-" +str(test_num) +".csv"
      filepath_stes = "./jrm_test/"+str(user_name+1)+"/face_test" + str(st) + "-" +str(test_num) +".csv"

      filepath_f = "./jrm_test/"+str(user_name+1)+"/factor_train" + str(st) + "-" +str(test_num) +".csv"
      filepath_s = "./jrm_test/"+str(user_name+1)+"/face_train" + str(st) + "-" +str(test_num) +".csv"
      t=int(_s*set_num*len(users)+user_name*set_num+test_num)

      _factor = np.loadtxt(filepath_f,delimiter=",")
      _factor_tes = np.loadtxt(filepath_ftes,delimiter=",")
      _signal = np.loadtxt(filepath_s,delimiter=",")
      f_var[:,t] = np.var(_factor,axis=0)
      s_var[:,t] = np.var(_signal,axis=0)
      f_test[:,t] = _factor_tes

      factor_vec[t,:]=_factor_tes.reshape(-1)
      #factor_vec[user_name*25+test_num,:,:]=_factor

pca = PCA(n_components=2).fit(factor_vec)
values = pca.transform(factor_vec)
print(values.shape)

#pred = KMeans(n_clusters=10).fit_predict(factor)
#clus = pred.reshape(9,-1)

#d = {'est_qual':est_q,'fact_var':f_var,'sig_var':s_var,'numb':num,'su_1':values[:,0],'su_2':values[:,1]}
d = {'est_qual':est_q,'numb':num,'su_1':values[:,0],'su_2':values[:,1]}
df= pd.DataFrame(data=d)

df_factor = pd.DataFrame(data=f_var.T)
df_face = pd.DataFrame(data=s_var.T)

df_all = pd.concat([df,df_factor,df_face,pd.DataFrame(data=f_test.T)],axis=1)

print(df_all.corr())
corr_arr = df_all.corr()
np.savetxt('corr_sheet_2.csv',corr_arr,delimiter=",")

from factor_analyzer import FactorAnalyzer

columns = df_all.columns
n=3
fa = FactorAnalyzer(n_factors=n,rotation="promax",impute="drop")
fa.fit(df_all.values)
print(fa.loadings_)
array=fa.loadings_
np.savetxt('analysis.csv',array)

#import seaborn as sns
#cm = sns.light_palette('red',as_cmap=True)
#df.style.background_gradient(cmap=cm) 
