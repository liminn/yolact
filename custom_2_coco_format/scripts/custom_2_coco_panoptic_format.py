import os
import cv2
import json
import glob
import random
import numpy as np
from pathlib import Path
from PIL import Image

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
    total_dict["info"]={"description": "COCO 2018 Panoptic Dataset", "url": "http://cocodataset.org", "version": "1.0", "year": 2020, "contributor": "https://arxiv.org/abs/1801.00868", "date_created": "2020-05-30 05:46:30.244442"}
    total_dict["licenses"] = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1, "name": "Attribution-NonCommercial-ShareAlike License"}]
    total_dict["images"] = []
    total_dict["annotations"] = []
    total_dict["categories"] = [{"supercategory": "text", "isthing": 1, "id": 1, "name": "text"}, 
                                {"supercategory": "background", "isthing": 0, "id": 2, "name": "background"}]
    return total_dict

# 获取描述image的dict
def get_coco_image_dict(image_path,image_id):
    file_name = os.path.split(image_path)[1]
    image = cv2.imread(image_path)
    height = image.shape[0]
    weight = image.shape[1]
    image_dict = {"license": 1, 
                "file_name": file_name, 
                "coco_url": "", 
                "height": height, 
                "width": weight, 
                "date_captured": "2020-07-16 17:02:52", 
                "flickr_url": "", 
                "id": image_id}
    return image_dict

# 获取描述annotation的dict
def get_coco_annotation_dict(image_path,image_id):
    file_name = os.path.split(image_path)[1]
    file_name = file_name.split(".jpg")[0] + ".png"
    seg_info_dict = {"segments_info":  [],
                    "file_name": file_name, "image_id": image_id}
    return seg_info_dict

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

def main(image_paths, label_panoptic_mask_save_dir, json_save_path,image_save_path):
    # 检查label_panoptic_mask_save_dir
    if not os.path.exists(label_panoptic_mask_save_dir):
        os.makedirs(label_panoptic_mask_save_dir)
    ids_list = []
    # 初始化total_dict
    total_dict = coco_instance_init()
    image_id = 0
    """
    # 天坑：COCOAPI希望读取dir+name，因此将不同src图片/mask整理至单独的文件下
    #      但是在本地数据中，不同文件夹里的图片文件名竟然相同，导致制作的src图片和panoptic mask因重名被替换
    #      因此，在训练过程中读取的panoptic mask和json文件的信息不匹配，一直报错。
    """
    base_name_list = [] 
    ignore_num = 0
    for image_path in image_paths:
        image = cv2.imread(image_path)
        base_name = os.path.split(image_path)[1]
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        # 赋予新的base_name和image_path
        if base_name in base_name_list:
            ignore_num +=1
            print("ignore_num:{}, base_name:{}".format(ignore_num, base_name))
            continue
            # base_name_new = base_name.split(".")[0] + base_name
            # base_name_list.append(base_name_new)
            # new_image_path = os.path.join(image_save_path, base_name_new)
            # cv2.imwrite(new_image_path, image)
        else:
            new_image_path = os.path.join(image_save_path, base_name)
            cv2.imwrite(new_image_path, image)
        base_name_list.append(base_name)
        # 获取image_dict
        image_dict= get_coco_image_dict(new_image_path,image_id)
        #print(image_dict)
        # 初始化描述图片annotation信息的seg_info_dict
        seg_info_dict = get_coco_annotation_dict(new_image_path,image_id)
        h,w,c = image.shape
        postfix = os.path.splitext(image_path)[-1]
        txt_path = image_path.split(postfix)[0]+".txt"
        # 防止图片无标注txt
        try:
            txt_file = open(txt_path,"r")
        except:
            print("no txt path:{}".format(image_path))
            continue
        # 将image_dict写入total_dict["images"]
        total_dict["images"].append(image_dict)
        # 绘制panoptic_mask(值为不重复的随机颜色)
        img_panoptic_mask = np.zeros((h, w, 3), np.uint8)
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
            bbox = [x,y,w,h]
            # 绘制panoptic_mask
            while 1:
                color_random_1 = [random.randint(1,255), random.randint(1,255), random.randint(1,255)]
                id_1 = color_random_1[2] + color_random_1[1]*256 + color_random_1[0]*256*256
                if id_1 not in ids_list:
                    break
            #print("color_random:{}, id:{}".format(color_random,id))
            ids_list.append(id_1)
            img_panoptic_mask = cv2.fillConvexPoly(img_panoptic_mask, points, color_random_1)
            # 向seg_info_dict添加该box的标注信息
            seg_info_dict["segments_info"].append({"id": id_1, "category_id": 1, "iscrowd": 0, "bbox": bbox, "area": area})
        while 1:
            color_random_2 = [random.randint(1,255), random.randint(1,255), random.randint(1,255)]
            id_2 = color_random_2[2] + color_random_2[1]*256 + color_random_2[0]*256*256
            if id_2 not in ids_list:
                break
        #print("color_random:{}, id:{}".format(color_random,id))
        index = img_panoptic_mask==[0,0,0]
        area2 = np.sum(index)
        # 向seg_info_dict添加背景的标注信息
        seg_info_dict["segments_info"].append({"id": id_2, "category_id": 2, "iscrowd": 0, "bbox": [0,0,int(image.shape[1]), int(image.shape[0])], "area": int(area2)})
        ids_list.append(id_2)
        temp = index * color_random_2 + (1-index) * img_panoptic_mask
        # 将seg_info_dict写入total_dict["annotations"]
        total_dict["annotations"].append(seg_info_dict)
        # 保存panoptic_mask
        base_name=os.path.split(new_image_path)[1]
        prefix = base_name.split(".jpg")[0]
        panoptic_mask_save_path = os.path.join(label_panoptic_mask_save_dir, prefix+".png")
        # if (panoptic_mask_save_path == "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/card_panoptic_20200717_4/card_panoptic_mask_train/4d8e6490-0598-4a4d-a7e4-efe16d86a67c_105809_IMG-105901-0010.png"):
        #     print(seg_info_dict)
        #     print(ids_list)
        #     break
        cv2.imwrite(panoptic_mask_save_path,temp)
        print("generating {}:{}".format(image_id, panoptic_mask_save_path))
        image_id += 1
    # 输出json文件
    with open(json_save_path,"w") as f:
        json.dump(total_dict,f)
        print("success")
    
    return 0
    
if __name__ == "__main__":
    
    # 定义原图路径(原图+标注txt)
    img_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/TEXT_DET/train"
    #img_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/TEXT_DET/valid"
    image_paths = get_all_image_path(img_dir)
    print(len(image_paths))
    # 定义json文件路径
    json_save_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/text_panoptic_20201113/text_panoptic_train.json"
    #json_save_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/text_panoptic_20201113/text_panoptic_valid.json"
    # 定义原图存储路径
    image_save_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/text_panoptic_20201113/text_panoptic_src_train"
    #image_save_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/text_panoptic_20201113/text_panoptic_src_valid"
    # 定义panoptic_mask存储路径
    label_panoptic_mask_save_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/text_panoptic_20201113/text_panoptic_mask_train"
    #label_panoptic_mask_save_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/text_panoptic_20201113/text_panoptic_mask_valid"
    
    main(image_paths, label_panoptic_mask_save_dir, json_save_path,image_save_path)

    """
    # test
    json_file = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/card_panoptic_20200718/panoptic/annotations/panoptic_train2017.json"
    #new_dict = json.loads(json_file)
    with open(json_file, 'r') as f:
        coco = json.load(f)
    coco['images'] = sorted(coco['images'], key=lambda x: x['id'])
    if "annotations" in coco:
        for img, ann in zip(coco['images'], coco['annotations']):
            assert img['file_name'][:-4] == ann['file_name'][:-4]
    for idx in range(len(coco['images'])-1):
        ann_info = coco['annotations'][idx]
        ann_path = Path("/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/card_panoptic_20200718/panoptic/panoptic_train2017") / ann_info['file_name']
        a = Image.open(ann_path)
        w, h = a.size
        masks = np.asarray(Image.open(ann_path), dtype=np.uint32)
        pixel_list = []
        ids = []
        for i in range(h):
            for j in range(w):
                pixel = list(masks[i,j,:])
                if pixel in pixel_list:
                    continue
                else:
                    pixel_list.append(pixel)
                    ids.append(pixel[0]+pixel[1]*256+pixel[2]*256*256)
        #print(pixel_list)
        anno_ids = []
        anno_segments_info = ann_info["segments_info"]
        #print(anno_segments_info)
        for i in range(len(anno_segments_info)):
            try:
                anno_ids.append(anno_segments_info[i]["id"])
            except:
                pass
        ids.sort()
        anno_ids.sort()
        
        # for i in ids:
        #     if i not in anno_ids:
        #         print("error: anno_ids:{}\n,ids:{}\n,anno_path:{}\n,pixel_lsit:{}".format(anno_ids,ids,ann_path,pixel_list))
        #         break
        if ids !=anno_ids:
            print("error: anno_ids:{}\n,ids:{}\n,anno_path:{}\n,pixel_lsit:{}".format(anno_ids,ids,ann_path,pixel_list))
            print(ann_info)
            assert ids == anno_ids
            #break
        else:
            print("right", idx,"\n",ann_path,"\n", ids,"\n", anno_ids)
    """
        
    """
    panoptic_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/card_panoptic_20200717_2/card_panoptic_mask_valid/5ab38ecc-f50b-43f0-b631-0ef607c943e8_110917_IMG-111118-0024.png"
    from PIL import Image
    masks = np.asarray(Image.open(panoptic_path), dtype=np.uint32)
    h,w,c = masks.shape
    pixel_list = []
    for i in range(h):
        for j in range(w):
            pixel = list(masks[i,j,:])
            if pixel in pixel_list:
                continue
            else:
                pixel_list.append(pixel)
    #[[52, 198, 235], [129, 174, 28], [240, 187, 173], [200, 101, 177], [200, 168, 183], [202, 210, 160], [100, 200, 205], [20, 228, 68]]
    print(pixel_list)
    result_dict = {}
    for i in pixel_list:
        result_dict[str(i)] = i[2] + i[1]*255 + i[0]*255*255
    print(result_dict)
    """
    





