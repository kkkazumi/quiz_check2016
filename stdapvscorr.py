# -*-coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as st

FACTOR_NUM = 10
TIME_NUM = 10

f_size = 50
name_list = ["inusan","kumasan","test119","test120","test121","tomato","torisan","usagisan"]
#name_list = ["inusan","kubosan","kumasan","nekosan","test119","test120","test121","tomato","torisan","usagisan","sarada"]
factor_list = ["time of trial","sucess rate","encourage behavior","sympathize behavior","karakai","not related","no reaction","total score","consecutive wins","consecutive loses"]

corr_mm = np.empty(len(name_list))

var = np.ones((len(name_list),len(factor_list)))
i = 0

for name in name_list:

  factor = np.loadtxt("./jiken/"+name+"/factor_before.csv",delimiter="\t")
  predicted = np.loadtxt("./jiken/"+name+"/m_pred.csv",delimiter=",")
  answered = np.loadtxt("./jiken/"+name+"/kibun_after.csv",delimiter=",")

  corr_mm[i] = abs(np.corrcoef(answered,predicted)[1,0])
  print name, corr_mm[i]

  for j in range(len(factor_list)):
    var[i,j] = np.std(factor[j])

  i=i+1


for j in range(len(factor_list)):
  plt.scatter(var[:,j],corr_mm,label=factor_list[j])
  val=st.pearsonr(corr_mm,var[:,j])
  print val
  #plt.title(factor_list[j])
  plt.legend()
  plt.xlabel('var')
  plt.xlim([0,1])
  plt.ylim([-1,1])
  plt.ylabel('corr')

  plt.show()
