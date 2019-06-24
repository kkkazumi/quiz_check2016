# -*-coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as st

def cos_sim(v1, v2):
  return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

FACTOR_NUM = 10
TIME_NUM = 10

f_size = 50
name_list = ["inusan","kumasan","test119","test120","test121","tomato","torisan","usagisan"]
#name_list = ["inusan","kubosan","kumasan","nekosan","test119","test120","test121","tomato","torisan","usagisan","sarada"]
factor_list = ["time of trial","sucess rate","encourage behavior","sympathize behavior","karakai","not related","no reaction","total score","consecutive wins","consecutive loses"]

#diff= np.empty(len(factor_after))
sim = np.empty(TIME_NUM)

var = np.ones((len(name_list),len(factor_list)))
i = 0


for name in name_list:
  factor_before = np.loadtxt("./jiken/"+name+"/factor_before.csv",delimiter="\t")
  factor_after = np.loadtxt("./jiken/"+name+"/factor_after.csv",delimiter="\t")
  predicted = np.loadtxt("./jiken/"+name+"/m_pred.csv",delimiter=",")
  answered = np.loadtxt("./jiken/"+name+"/kibun_after.csv",delimiter=",")

  for i in range(len(factor_after)):
    for j in range(len(factor_before)):
      sim[i]=cos_sim(factor_before[j,:],factor_after[i])

  diff = predicted - answered

  print(name,st.pearsonr(sim,diff))

  plt.scatter(sim,diff)
  plt.title(name)
  plt.xlabel('cos_sim')
  plt.xlim([0,1])
  plt.ylim([-1,1])
  plt.ylabel('diff pred - ans')

  plt.show()
