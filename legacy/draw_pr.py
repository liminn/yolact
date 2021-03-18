import time
import cv2
import torch
import os
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import random
from pathlib import Path
from utils import timer
from yolact import Yolact
from utils.augmentations import FastBaseTransform
from layers.output_utils import postprocess
from data import MEANS, COLORS
from collections import defaultdict
from data import cfg, set_cfg, set_dataset
import shapely
import numpy as np
from shapely.geometry import Polygon, MultiPoint  # 多边形

def bbox_iou_eval(box1, box2):
    '''
    利用python的库函数实现非矩形的IoU计算
    :param box1: list,检测框的四个坐标[x1,y1,x2,y2,x3,y3,x4,y4]
    :param box2: lsit,检测框的四个坐标[x1,y1,x2,y2,x3,y3,x4,y4]
    :return: IoU
    '''
    box1 = np.array(box1).reshape(4, 2)  # 四边形二维坐标表示
    # python四边形对象，会自动计算四个点，并将四个点从新排列成
    # 左上，左下，右下，右上，左上（没错左上排了两遍）
    poly1 = Polygon(box1).convex_hull
    box2 = np.array(box2).reshape(4, 2)
    poly2 = Polygon(box2).convex_hull

    if not poly1.intersects(poly2):  # 若是两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            iou = float(inter_area) / (poly1.area + poly2.area - inter_area)
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0

    return iou

def get_min_rect(image):
    image = image[:,:,np.newaxis]*255
    #image = np.concatenate((image, image, image), axis=-1).astype(np.uint8)
    image = np.array(image,np.uint8)
    #print(image.shape)
    #cv2.imwrite('test/11.png',image)
    contours, hier = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minAreaRect = cv2.minAreaRect(contours[0])
    rectCnt = np.int64(cv2.boxPoints(minAreaRect))
    cv2.drawContours(image, [rectCnt], 0, (128), 3)
    a = random.randint(1,10)
    cv2.imwrite('test/{}.png'.format(a),image)
    #print(points)
    return rectCnt

def cal_tp_fp_fn_single(net,image_path,txt_path,iou_threshold = 0.5, score_threshold=0.15):
    tp = 0  # 分类正确，且与GTiou大于等于iou_threshold
    fp = 0  # 分类不正确，或与GTiou小于iou_threshold
    fn = 0  # GT数减去tp数
    classes, masks = evalimage(net, image_path,score_threshold)
    boxs_gt = get_gt_info(txt_path)
    total = len(boxs_gt)
    #print(len(boxs_gt),boxs_gt[0])
    boxs_prd = []
    #print(len(classes))
    for i in range(len(masks)):
        points = get_min_rect(masks[i].detach().cpu().numpy())
        boxs_prd.append(points)
    bool_tp = np.zeros(len(boxs_prd)).astype(np.uint8)
    bool_gt_used = np.zeros(len(boxs_gt)).astype(np.uint8)
    #print(len(boxs_prd),boxs_prd,boxs_prd[0],boxs_prd[1])
    #print(len(boxs_prd))

    for i in range(len(boxs_prd)):
        for j in range(len(boxs_gt)):
            if(bool_gt_used[j]==1): # 保证了一个gt只能匹配到一次，但没有保证让最高iou预测box的那个匹配，但是应该不影响计算结果
                continue
            box1 = [boxs_prd[i][0][0], boxs_prd[i][0][1], boxs_prd[i][1][0], boxs_prd[i][1][1],boxs_prd[i][2][0], boxs_prd[i][2][1], boxs_prd[i][3][0], boxs_prd[i][3][1]]   # 左上，右上，右下，左下
            box2 = [boxs_gt[j][0][0], boxs_gt[j][0][1], boxs_gt[j][1][0], boxs_gt[j][1][1], boxs_gt[j][2][0], boxs_gt[j][2][1], boxs_gt[j][3][0], boxs_gt[j][3][1]]
            iou = bbox_iou_eval(box1, box2)
            #print(i,iou)
            if iou >= iou_threshold and classes.cpu().numpy()[i]==0:
                bool_tp[i]=1
                bool_gt_used[j]=1
    tp = np.sum(bool_tp)
    fp = len(boxs_prd)-tp
    fn = len(boxs_gt)-tp
    #print((tp,fp,fn))
    return (tp,fp,fn)

def evalimage(net:Yolact, path:str, score_threshold=0.15):
    frame = torch.from_numpy(cv2.imread(path)).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)
    h, w, _ = frame.shape
    classes, scores, boxes, masks = postprocess(preds, w, h, visualize_lincomb = False,
                                crop_masks  = True,
                                score_threshold  = score_threshold)
    """
    print(classes.shape)  # [num_det]
    print(classes.cpu().numpy())        # [num_det]
    print("=============")
    print(len(scores),scores[0].shape)    # [num_det,2] box的score和mask的score
    print(scores)          # 列表，列表第一项，存储所有box的box的score，列表第二项存储所有box的mask的score，最终得分是两者相乘
    print(scores[1][0].cpu().numpy())
    print("=============")
    print(boxes.shape)    # [num_det, 4] (x_min,y_min,x_max,y_max)
    print(boxes.cpu().numpy())  
    print("=============")
    print(masks.shape)    # [num_det, h, w] ,对于每个实例，产生0/1的mask
    print(0 in masks[0].cpu().detach().numpy())
    print(1 in masks[0].cpu().detach().numpy())
    print(2 in masks[0].cpu().detach().numpy())
    """
    #cv2.imwrite(save_path, img_numpy)
    return classes, masks
    
def get_gt_info(txt_path):
    try:
        txt_file = open(txt_path,"r")
    except:
        print("error:{}".format(txt_file))
    lines = txt_file.readlines()
    boxs = []
    for line in lines:
        s = line.strip().split(",")
        point_1 = [round(float(s[1])), round(float(s[2]))]
        point_2 = [round(float(s[3])), round(float(s[4]))]
        point_3 = [round(float(s[5])), round(float(s[6]))]
        point_4 = [round(float(s[7])), round(float(s[8]))]
        box=[point_1,point_2,point_3,point_4]
        boxs.append(box)
    return boxs

def evalimages(net:Yolact, input_folder:str):
    for p in Path(input_folder).glob('*.jpg'): 
        path = str(p)
        #print(path)
        evalimage(net,path)
    #print('Done.')
    return 0

# 获取目录下所有图片的路径
def get_all_image_path(dir):
    image_paths = []
    for path,d,filelist in os.walk(dir):
        for filename in filelist:
            if(filename.endswith("jpg")):
                image_paths.append((os.path.join(path, filename)))
    return image_paths

if __name__ == "__main__":
    cudnn.fastest = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    dir = "/home/dell/zhanglimin/code/instance_seg/yolact-master/test"
    trained_model = "weights/card_weight_yolact++_20200602/yolact_plus_base_93_100000.pth"
    config_name = "yolact_plus_base_config"
    set_cfg(config_name)
    net = Yolact()
    net.load_weights(trained_model)
    net.eval()
    net = net.cuda()
    net.detect.use_fast_nms = True
    net.detect.use_cross_class_nms = False
    cfg.mask_proto_debug = False
    evalimages(net, dir)
    #print("xxxxxxxxx")
    txt_path = "/home/dell/zhanglimin/code/instance_seg/yolact-master/test/0e20cae9-7bae-47bc-b8e0-4eeae97057fe_370199315.txt"
    boxs = get_gt_info(txt_path)
    #print(len(boxs))
    #print(boxs)

    #image_path = "/home/dell/zhanglimin/code/instance_seg/yolact-master/test/0e20cae9-7bae-47bc-b8e0-4eeae97057fe_370199315.jpg"
    #tp,fp,fn = cal_tp_fp_fn_single(net,image_path,txt_path)

    # 待遍历的文件夹路径
    dir = "/home/dell/zhanglimin/data/card/valid/v2_export_142_2020-05-14_2020-05-17/det/ori_image"
    image_paths = get_all_image_path(dir)
    
    score_list = np.linspace(0.0,1.0,100)
    result_lsit = []
    for i in range(len(score_list)):
        tp_total = 0
        fp_total = 0
        fn_total = 0
        precision = 0.0
        recall = 0.0
        for image_path in image_paths:
            #print(image_path)
            score_threshold = score_list[i]
            txt_path = image_path.split(".jpg")[0]+".txt"
            tp,fp,fn = cal_tp_fp_fn_single(net,image_path,txt_path,iou_threshold = 0.5, score_threshold=score_threshold)
            tp_total += tp
            fp_total += fp
            fn_total += fn
        precision = tp_total/(tp_total+fp_total)
        recall = tp_total/(tp_total+fn_total)
        line = str(score_threshold)+","+str(precision)+","+str(recall)+"\n"
        f = open("/home/dell/zhanglimin/code/instance_seg/yolact-master/pr.txt",'a')
        f.write(line)
        print("score_threshold:{},precision:{},recall:{}".format(score_threshold,precision,recall))
        #result_lsit.append([score_threshold,precision,recall])

    

    



