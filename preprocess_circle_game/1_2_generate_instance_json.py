"""check line label, and get line mask"""
import os
import cv2
import glob
import json
import math
import random
import pycocotools
import numpy as np 
from pycocotools import mask as maskUtils
from pycocotools import coco as COCO

### 初始化coco dict
def coco_instance_init():
    total_dict = {}
    total_dict["info"]={"description": None, "url": None, "version": None, "year": 2020, "contributor": None, "date_created": "2020-05-30 05:46:30.244442"}
    total_dict["licenses"] = [{"url": None, "id": 0, "name": None}]
    total_dict["images"] = []
    total_dict["type"] = "instances"
    total_dict["annotations"] = []
    total_dict["categories"] = [{"supercategory": "circle", "isthing": 1, "id": 1, "name": "circle"}]
    return total_dict
    
### 获取描述image的dict
def get_coco_image_dict(image_path,image_id):
    file_name = os.path.split(image_path)[1]
    image = cv2.imread(image_path)
    height = image.shape[0]
    weight = image.shape[1]
    image_dict = {"license": 0, "url": None, "file_name": file_name, "height": height, "width": weight, "date_captured": None, "id": image_id} 
    return image_dict

### 获取描述instance的dict
def get_coco_instance_dict(image_id, instance_id, segmentation, area, bbox, rle, h, w, RLE=False):
    if not RLE:
        instance_dict = {"id": instance_id, "image_id": image_id, "category_id": 1, "segmentation": segmentation, 
                    "area": area, "bbox": bbox, "iscrowd": 0}
    else:
        instance_dict = {"id": instance_id, "image_id": image_id, "category_id": 1, "segmentation": rle, # rle is {"counts": rle, "size": [h, w]}
                    "area": area, "bbox": bbox, "iscrowd": 0}
    return instance_dict


def generate_instance_json(label_json_path, query_image_dir, image_save_dir, json_save_path, ignore_image_list):
    
    ### 初始化coco dict
    image_id = 0
    instance_id = 0
    total_dict = coco_instance_init()
    RLE = True

    if isinstance(label_json_path, list): # 如果是列表
        for item in label_json_path: # 逐个遍历
            with open(item) as f:
                json_dict_list = json.load(f)
                for i in range(len(json_dict_list)):
                    item = json_dict_list[i]
                    data_id = item["dataId"]
                    ### get query image path
                    image_info = item["imageInfo"]["path"]
                    name = os.path.basename(image_info)
                    if name in ignore_image_list:
                        print("ignore:", name)
                        continue
                    image_path = query_image_dir + image_info
                    #print(data_id, image_path)
                    if not os.path.exists(image_path):
                        continue
                    try:
                        point_dict_list = item["label"]["smaple-class"]
                    except:
                        continue
                    ### get points_list in query image
                    points_list = []
                    for j in range(len(point_dict_list)):
                        point_item = point_dict_list[j]["shape"]["geometry"]
                        #print(point_item)
                        points_list_temp = []
                        for point in point_item:
                            points_list_temp.append([point["x"], point["y"]])
                        points_list.append(points_list_temp)
                        ### read query image
                    image = cv2.imread(image_path)
                    ### 先缩放图像，长边为550
                    h,w,c = image.shape
                    max_size = 550
                    if h > w:
                        h_new = max_size
                        w_new = w * h_new/h
                    else:
                        w_new = max_size
                        h_new = h * w_new/w
                    image = cv2.resize(image, (int(w_new), int(h_new)))
                    h_ratio = h_new/h
                    w_ratio = w_new/w

                    ### draw points on mask
                    for k in range(len(points_list)):
                        list_temp = points_list[k]
                        #print(list_temp)
                        mask_grey = np.zeros((image.shape[0], image.shape[1]))
                        for l in range(len(list_temp)-1):
                            start_point = list_temp[l]
                            end_point = list_temp[l+1]
                            #print(start_point)
                            #thickness = random.randint(3,4)
                            thickness = 7 #8 #6 #4
                            cv2.line(mask_grey, (int(start_point[0]*w_ratio), int(start_point[1]*h_ratio)), (int(end_point[0]*w_ratio), int(end_point[1]*h_ratio)), (255), thickness)

                        ### 依据mask，获取bbox/segmentation(rle)/areas
                        mask_grey_copy = mask_grey.astype(np.uint8).copy()
                        ret, binary = cv2.threshold(mask_grey_copy,1,255,cv2.THRESH_BINARY)  
                        contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
                        areas = [cv2.contourArea(c) for c in contours] 
                        segmentation = None
                        area = None
                        bbox = None
                        ### Find the index of the largest contour
                        max_index = np.argmax(areas)  
                        cnt=contours[max_index]
                        ### 获取bbox
                        x,y,w,h = cv2.boundingRect(cnt)
                        bbox = [x,y,w,h]
                        ### 获取segmentation
                        segmentation = []
                        for c in contours:
                            segmentation_temp = []
                            for point in c:
                                point = point[0]
                                #print(point)
                                segmentation_temp += [float(point[0]), float(point[1])] # 确认segmentation希望是xy还是yx，确认contour返回是xy还是yx
                            segmentation.append(segmentation_temp)
                            #print(segmentation)
                        ###获得area
                        mask_temp = mask_grey.copy()
                        mask_temp[mask_temp<1] = 0
                        mask_temp[mask_temp>=1] = 1
                        area = float(np.sum(mask_temp))
                        ### 获取mask的RLE表示
                        rle = None
                        bin_mask = np.asfortranarray(mask_grey) 
                        bin_mask[bin_mask<1] = 0
                        bin_mask[bin_mask>=1] = 1
                        ### ref: https://github.com/dbolya/yolact/issues/544
                        rle = pycocotools.mask.encode(np.asfortranarray(bin_mask.astype(np.uint8)))
                        rle['counts'] = rle['counts'].decode('ascii')  # json.dump doesn't like bytes strings

                        ### 获取描述instance的dict，并添加进total_dict
                        h,w,c = image.shape
                        instance_dict= get_coco_instance_dict(image_id, instance_id, segmentation, area, bbox, rle, h, w, RLE)
                        assert 1<=len(instance_dict["segmentation"])<=2
                        total_dict["annotations"].append(instance_dict)
                        instance_id += 1 # 更新instance_id

                    ###获取描述image(合成之后)的dict，并添加进total_dict
                    name = os.path.split(image_path)[1]
                    #image_save_path = os.path.join(image_save_dir, str(data_id) + "_" + name)
                    image_save_path = os.path.join(image_save_dir, str(image_id) + "_" + name)
                    cv2.imwrite(image_save_path, image)
                    print("{}/{}, save image:{}".format(image_id, len(json_dict_list), image_save_path))
                    image_dict= get_coco_image_dict(image_save_path,image_id)
                    total_dict["images"].append(image_dict)
                    image_id += 1 # 更新image_id

    ### 保存total_dict至json文件
    #json_save_path = os.path.join(json_save_dir, json_name)
    with open(json_save_path,"w") as f:
        #print(total_dict)
        json.dump(total_dict,f)
        print("dump:", json_save_path)

if __name__ == "__main__":
    ### define input path
    #"""
    ### train
    label_json_path = ["/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0/circle_new.json",
                        "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0/20210202_800_crop.json"]
    query_image_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0"
    ignore_image_list = ["IMG-125111-001.jpg"]
    #"""
    """
    ### test
    label_json_path = ["/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/test_20200127.json",
                       "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210130_446_crop.json"]
    query_image_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0"
    ignore_image_list = []
    """

    ### define image/json save dir
    image_save_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0/20210205_thickness7_train"
    if(not os.path.exists(image_save_dir)):
        os.makedirs(image_save_dir)
    json_save_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0/20210205_thickness7_train.json"
    
    ### run
    generate_instance_json(label_json_path, query_image_dir, image_save_dir, json_save_path, ignore_image_list)
