
# -*- coding: UTF-8 -*-
import os
import sys
import cv2
import random
import argparse
from joint_cards import JointCards

def get_bbox_info(txt_path):
    f = open(txt_path)
    lines = f.readlines()
    bbox_infos = []
    for i, line in enumerate(lines):
        line = line.strip().split(",")
        char = line[0]
        x1 = float(line[1])
        y1 = float(line[2])
        x2 = float(line[3])
        y2 = float(line[4])
        x3 = float(line[5])
        y3 = float(line[6])
        x4 = float(line[7])
        y4 = float(line[8])
        bbox_infos.append([char, x1, y1, x2, y2, x3, y3, x4, y4])
    return bbox_infos

def visulaize_results(image_path, bbox_infos, box_cluster):
    image = cv2.imread(image_path)
    for i in range(len(box_cluster)):
        cluster_list = box_cluster[i]
        random_color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        for j in range(len(cluster_list)):
            bbox_info = bbox_infos[cluster_list[j]]
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox_info[1:]
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x3), int(y3)), random_color, 3)
    path, base_name = os.path.split(image_path)
    save_path = os.path.join(path, base_name.split(".jpg")[0]+"_result.jpg")
    cv2.imwrite(save_path, image)


if __name__ == "__main__":
    """
    python3 demo.py --txt xx/test_images/test_1.txt --image xx/test_images/test_1.jpg
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', type=str, help='path to txt')
    parser.add_argument('--image', type=str, help='path to image')
    args = parser.parse_args()
    
    '''1.输入数据格式：[[char, x1, y1, x2, y2, x3, y3, x4, y4],[]...,[]]'''
    bbox_infos = get_bbox_info(args.txt)
    '''2.实例化JointCards类'''
    J = JointCards(bbox_infos)
    '''3.获取卡片聚类结果'''
    box_cluster = J.get_results()
    '''4.输出数据格式：[[0, 1], [2, 3]]'''
    print(box_cluster)
    '''5.可视化结果'''
    visulaize_results(args.image, bbox_infos, box_cluster)
