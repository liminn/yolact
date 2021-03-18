"""check line label, and get line mask"""
import os
import cv2
import glob
import json
import math
import random
import numpy as np 

def label_to_mask(label_json_path, query_image_dir, merge_save_dir):
    if not os.path.exists(merge_save_dir):
        os.makedirs(merge_save_dir)
    
    with open(label_json_path) as f:
        # the json file is a list, each list item is a dict
        json_dict_list = json.load(f)
        #print(len(json_dict_list))
        
        for i in range(len(json_dict_list)):
            item = json_dict_list[i]
            #print(item)
            #print(item.keys()) # dict_keys(['dataId', 'fileName', 'imageUrl', 'imageInfo', 'label', 'uploader', 'lastModified'])
            ### get data id
            data_id = item["dataId"]
            ### get query image path
            image_info = item["imageInfo"]["path"]
            name = os.path.basename(image_info)
            #print(name)
            #image_path = glob.glob(query_image_dir + "*/*/"+name)[0]
            image_path = query_image_dir + image_info
            print(data_id, image_path)
            #try:
            ### ignore not-know image 
            #label_class = item["label"]["class"]
            # line_query_crop_1/newest_handine_2020-11-25.json将空样本标成了"not-know",而不是negtive，此处关闭
            #if label_class == "not-know": # "not-know"/"negative"/"positive"
            #    print("ignore not-know image ")
            #    continue
            try:
                point_dict_list = item["label"]["smaple-class"]
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
                ### draw points on mask
                mask_grey = np.zeros((image.shape[0], image.shape[1]))
                for k in range(len(points_list)):
                    list_temp = points_list[k]
                    #print(list_temp)
                    for l in range(len(list_temp)-1):
                        start_point = list_temp[l]
                        end_point = list_temp[l+1]
                        #print(start_point)
                        #thickness = random.randint(3,4)
                        thickness = 5
                        cv2.line(mask_grey, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])), (255), thickness)
            except:
                ### read query image
                image = cv2.imread(image_path)
                if image is None: 
                    continue
                ### draw points on mask
                mask_grey = np.zeros((image.shape[0], image.shape[1]))
                print("no label, mask is black")
            #print(point_dict_list)
            ### merge mask and query image
            image_green = cv2.merge([np.zeros((image.shape[0], image.shape[1])), np.ones((image.shape[0], image.shape[1]))*255, np.zeros((image.shape[0], image.shape[1]))])
            mask_alpha = mask_grey[:, :, np.newaxis]/255.0*0.5
            mask_alpha = cv2.merge([mask_alpha, mask_alpha, mask_alpha])
            image_merge = image_green * mask_alpha + (1-mask_alpha)*image
            image_merge = np.hstack([image, image_merge])

            ### save query_image/mask/merge_results
            #image_save_path = os.path.join(image_save_dir, str(data_id) + "_" + name)
            #mask_grey_save_path = os.path.join(mask_save_dir, name.split(".jpg")[0]+".png")
            image_merge_save_path = os.path.join(merge_save_dir, str(data_id) + "_" + name.split(".jpg")[0]+".png")
            #cv2.imwrite(image_save_path, image)
            #cv2.imwrite(mask_grey_save_path, mask_grey)
            cv2.imwrite(image_merge_save_path, image_merge)
            #print("save: {}".format(image_merge_save_path))
            print(i,"/", len(json_dict_list))
            

if __name__ == "__main__":
    ### define input path
    #label_json_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0/circle_new.json"
    #query_image_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0"
    # label_json_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0/20210202_800_crop.json"
    # query_image_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0"

    #label_json_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/test_20200127.json"
    #query_image_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0"
    label_json_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210130_446_crop.json"
    query_image_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0"
    
    ### define save path
    merge_save_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210130_446_crop_label_check"
    
    ### run
    label_to_mask(label_json_path, query_image_dir, merge_save_dir)
