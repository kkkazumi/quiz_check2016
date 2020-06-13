import numpy as np
import cv2

EMO_NUM = 4
SIT_NUM = 10
USR_NUM = 9

xlabel = ["hap","sup","ang","sad"]

str_func = ["time of trial","rate of wins","encourage","symp","teasing","unrelated","no actions","total points","consecutive wins","consecutive losses"]
str_emo = ["hap","sup","ang","sad"]

im = np.zeros((EMO_NUM,1200,1600,3))
print(im.shape)
total = np.zeros((2400,3200,3))

for sit_num in range(SIT_NUM):
  for emo_num in range(EMO_NUM):

    im[emo_num,:,:,:] = cv2.imread("./diff_graph/sit-"+str_func[sit_num]+"_emo-"+str_emo[emo_num]+".png")

  total[:1200,:1600,:] = im[0,:,:,:]
  total[:1200,1600:,:] = im[1,:,:,:]
  total[1200:,:1600,:] = im[2,:,:,:]
  total[1200:,1600:,:] = im[3,:,:,:]
  cv2.imwrite("./diff_graph/data"+str_func[sit_num]+".png",total)
  #cv2.waitKey(0)
  #cv2.destroyAllWindows()

