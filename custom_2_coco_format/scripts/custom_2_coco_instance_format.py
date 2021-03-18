import os
import cv2
import json
import glob
import numpy as np

# 计算多边形的面积
def cal_poly_area(points):
    """方法一"""
    # img = np.zeros((w, h), np.uint8)
    # x = cv2.fillConvexPoly(img, points, (255))
    # area = np.sum(x==255)
    """方法二"""
    area = cv2.contourArea(points,True)
    return area

# 计算多边形的正外接矩形，返回(左上角x,左上角y,宽，高)
def cal_bounding_rect(points):
    x,y,w,h = cv2.boundingRect(points)
    return (x,y,w,h)

# 获取目录下所有图片的路径
def get_all_image_path(dir):
    image_paths = []
    for path,d,filelist in os.walk(dir):
        for filename in filelist:
            if(filename.endswith("jpg") or filename.endswith("jpeg") or filename.endswith("png") or filename.endswith("JPG") or filename.endswith("JPEG") or filename.endswith("PNG")):
                image_paths.append((os.path.join(path, filename)))
    return image_paths

# 初始化coco dict
def coco_instance_init():
    total_dict = {}
    total_dict["info"]={"description": None, "url": None, "version": None, "year": 2020, "contributor": None, "date_created": "2020-05-30 05:46:30.244442"}
    total_dict["licenses"] = [{"url": None, "id": 0, "name": None}]
    total_dict["images"] = []
    total_dict["type"] = "instances"
    total_dict["annotations"] = []
    total_dict["categories"] = [{"supercategory": "text", "isthing": 1, "id": 1, "name": "text"}]
    return total_dict
    
# 获取描述image的dict
def get_coco_image_dict(image_path,image_id,base_dir):
    #file_name = image_path.split(base_dir+"/")[1]
    file_name = os.path.split(image_path)[1]
    image = cv2.imread(image_path)
    height = image.shape[0]
    weight = image.shape[1]
    image_dict = {"license": 0, "url": None, "file_name": file_name, "height": height, "width": weight, "date_captured": None, "id": image_id} 
    return image_dict

# 获取描述instance的dict
def get_coco_instance_dict(image_id, instance_id, segmentation, area, bbox):
    instance_dict = {"id": instance_id, "image_id": image_id, "category_id": 1, "segmentation": segmentation, 
                    "area": area, "bbox": bbox, "iscrowd": 0}
    return instance_dict
    
if __name__ == "__main__":
    # 待遍历的文件夹路径
    ### card
    #dir = "/media/dell/data/det/valid"
    #dir = "/media/dell/data/det/train"
    ### text
    #dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/TEXT_DET/train"
    dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/TEXT_DET/valid"
    # coco文件的保存路径
    ### card
    #json_save_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/card_instance_train.json"
    ### text
    #json_save_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/text_panoptic_20201113/text_instance_train.json"
    json_save_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/text_panoptic_20201113/text_instance_valid.json"
    
    image_paths = get_all_image_path(dir)
    image_id = 0
    instance_id = 0
    # 初始化dict
    total_dict = coco_instance_init()
    base_name_list = [] 
    ignore_num = 0
    for image_path in image_paths:
        base_name = os.path.split(image_path)[1]
        if base_name in base_name_list:
            ignore_num +=1
            print("ignore_num:{}, base_name:{}".format(ignore_num, base_name))
            continue
        base_name_list.append(base_name)
        print(image_path)
        # 获取描述image的dict
        image_dict= get_coco_image_dict(image_path,image_id,dir)
        #print(image_dict)
        total_dict["images"].append(image_dict)
        postfix = os.path.splitext(image_path)[-1]
        txt_path = image_path.split(postfix)[0]+".txt"
        # 防止图片无标注文件
        try:
            txt_file = open(txt_path,"r")
        except:
            continue
        # 读取标注信息
        lines = txt_file.readlines()
        for line in lines:
            s = line.strip().split(",")
            try:
                # point_1 = [round(float(s[1])), round(float(s[2]))]
                # point_2 = [round(float(s[3])), round(float(s[4]))]
                # point_3 = [round(float(s[5])), round(float(s[6]))]
                # point_4 = [round(float(s[7])), round(float(s[8]))]
                point_1 = [round(float(s[0])), round(float(s[1]))]
                point_2 = [round(float(s[2])), round(float(s[3]))]
                point_3 = [round(float(s[4])), round(float(s[5]))]
                point_4 = [round(float(s[6])), round(float(s[7]))]
            except:
                print(txt_path)
                continue
            points = np.array([[point_1,point_2,point_3,point_4]])
            # 计算多边形的面积
            area = cal_poly_area(points)
            # 计算多边形的正外接矩形，返回(左上角x,左上角y,宽，高)
            x,y,w,h = cal_bounding_rect(points)
            # cv2.fillConvexPoly(img, points, (255))
            # cv2.rectangle(img, (x,y), (x+w,y+h), 255, 2)
            # cv2.imwrite("/Users/bron/Desktop/2.png",img)
            segmentation = [point_1+point_2+point_3+point_4]
            bbox = [x,y,w,h]
            # 获取描述instance的dict
            instance_dict= get_coco_instance_dict(image_id, instance_id, segmentation, area, bbox)
            instance_id += 1
            #print(instance_dict)
            total_dict["annotations"].append(instance_dict)
        image_id += 1
    # 输出json文件
    with open(json_save_path,"w") as f:
        json.dump(total_dict,f)
        print("success")
    








