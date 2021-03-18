import numpy as np
import matplotlib.pyplot as plt

#plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 
#plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

txt_file = open("/home/dell/zhanglimin/code/instance_seg/yolact-master/pr.txt","r")
lines = txt_file.readlines()
score_lsit = []
precision_lsit = []
recall_lsit = []
for line in lines:
    score = float(line.split(",")[0])
    precison = float(line.split(",")[1])
    recall = float(line.strip().split(",")[2])
    score_lsit.append(score)
    precision_lsit.append(precison)
    recall_lsit.append(recall)
score_arr = np.array(score_lsit)
precision_arr = np.array(precision_lsit)
recall_arr = np.array(recall_lsit)
score_arr = np.around(score_arr, decimals=2)
precision_arr = np.around(precision_arr, decimals=3)
recall_arr = np.around(recall_arr, decimals=3)
#print(score_arr[0])
#print(precision_arr)
#print(recall_arr)

x = recall_arr
y = precision_arr
print(len(x),x[0],x[1])
print(len(y),y[0],y[1])
plt.plot(x,y, marker='o',color='red',label='score_threshold')
plt.title("PR curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
#遍历每一个点，使用text将y值显示
for i in range(len(x)):
    #print(i)
    #x_ = x[i],
    #y_ = y[i]
    #score_ = str(score_arr[i])
    plt.text(x[i],y[i],str(score_arr[i]),fontsize=5)

plt.legend()
plt.show()
plt.savefig("/home/dell/zhanglimin/code/instance_seg/yolact-master/pr_100.png", dpi=200)




