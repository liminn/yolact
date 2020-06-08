
# -*- coding: UTF-8 -*-
import os
import sys
import cv2
import glob
import random
import argparse
from joint_cards import get_results

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

def visulaize_results(image_path, cluster_rectangle):
    image = cv2.imread(image_path)
    for i in range(len(cluster_rectangle)):
        rectangle = cluster_rectangle[i]
        left_top = (rectangle[0],rectangle[1])
        right_bottom = (rectangle[4],rectangle[5])
        random_color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        image = cv2.rectangle(image, left_top, right_bottom, random_color, 3)
    path, base_name = os.path.split(image_path)
    save_path = os.path.join(path, base_name.split(".jpg")[0]+"_result.jpg")
    cv2.imwrite(save_path, image)

if __name__ == "__main__":
    """
    Usage: python3 demo.py --path xx/test_resources
    """
    # 获取测试资源路径
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to test resources')
    args = parser.parse_args()
    test_resource = args.path
    txt_paths = glob.glob(os.path.join(test_resource,"*.txt"))
    txt_paths.sort(reverse = False)

    # 遍历所有的测试txt文件，逐个进行拼接
    for i in range(len(txt_paths)):
        txt_path = txt_paths[i]
        image_path = txt_path.split(".txt")[0]+".jpg"
        # 1.制作标准输入数据格式：[[char, x1, y1, x2, y2, x3, y3, x4, y4],[]...,[]]
        bbox_infos = get_bbox_info(txt_path)
        # 2.获取拼接结果：{'index': [[0], [1, 2]], 'bbox': [[583, 233, 737, 233, 737, 354, 583, 354], [546, 389, 811, 389, 811, 468, 546, 468]], 'text': ['h', 'Zb']}
        joint_dict = get_results(bbox_infos)
        print("input: {}\noutput: {}\n".format(os.path.basename(txt_path), joint_dict))
        # 3.可视化拼接结果：在原图上，绘制出各区域的矩形框
        visulaize_results(image_path, joint_dict["bbox"])
