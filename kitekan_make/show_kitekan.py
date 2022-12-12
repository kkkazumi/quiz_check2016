import cv2
import numpy as np

USER_NUM = 9
FACTOR_NUM = 10
FACE_TYPE = 4

print("input plot directory name(without any /)")
dirname=input()

test_im=cv2.imread("./"+dirname+"/graph_u2_f4_s3.png")
height,width=test_im.shape[:2]
#_im=np.zeros_like(test_im)
#im=np.tile(_im,9)
#print(im.shape)
#input()
print("note that save mode doesnt work, and make total/ dir")
print("input show graph or save graph: save/show")
flg=input()

for factor_type in range(FACTOR_NUM):
  for signal_type in range(FACE_TYPE):

    im1=cv2.imread("./"+dirname+"/graph_u1_f"+str(factor_type)+"_s"+str(signal_type)+".png")
    im2=cv2.imread("./"+dirname+"/graph_u2_f"+str(factor_type)+"_s"+str(signal_type)+".png")
    im3=cv2.imread("./"+dirname+"/graph_u3_f"+str(factor_type)+"_s"+str(signal_type)+".png")
    im4=cv2.imread("./"+dirname+"/graph_u4_f"+str(factor_type)+"_s"+str(signal_type)+".png")
    im5=cv2.imread("./"+dirname+"/graph_u5_f"+str(factor_type)+"_s"+str(signal_type)+".png")
    im6=cv2.imread("./"+dirname+"/graph_u6_f"+str(factor_type)+"_s"+str(signal_type)+".png")
    im7=cv2.imread("./"+dirname+"/graph_u7_f"+str(factor_type)+"_s"+str(signal_type)+".png")
    im8=cv2.imread("./"+dirname+"/graph_u8_f"+str(factor_type)+"_s"+str(signal_type)+".png")
    im9=cv2.imread("./"+dirname+"/graph_u9_f"+str(factor_type)+"_s"+str(signal_type)+".png")

    im_h1=cv2.hconcat([im1,im2,im3])
    im_h2=cv2.hconcat([im4,im5,im6])
    im_h3=cv2.hconcat([im7,im8,im9])
    im_all = cv2.vconcat([im_h1,im_h2,im_h3])
    save_fig=cv2.resize(im_all,dsize=(width*2,height*2))
    title="f"+str(factor_type)+"_s"+str(signal_type)

    if(flg=="save"):
      savefig_filename="./"+dirname+"/total/"+title+".png"
      cv2.imwrite(savefig_filename,save_fig)
    elif(flg=="show"):
      cv2.imshow(title,save_fig)
      cv2.waitKey(0)
      cv2.destroyWindow(title)
