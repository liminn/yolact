import os
import cv2
import numpy as np
from pycocotools.coco import COCO

COCO_LABEL_MAP = { 1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,
                   9:  9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}

class COCOAnnotationTransform(object):
    
    def __init__(self):
        self.label_map = COCO_LABEL_MAP
        
    def __call__(self, target, width, height):
        """
        Args:
            target (dict): 包含某张图片的所有annotation_dict的列表
            height (int): 图片的高
            width (int): 图片的宽
        Returns:
            [[xmin, ymin, xmax, ymax, class_idx], ...], xmin, ymin, xmax, ymax is normalized(0~1)
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                label_idx = obj['category_id']
                if label_idx >= 0:
                    # 疑问：为什么减去1？
                    label_idx = self.label_map[label_idx] - 1
                final_box = list(np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, class_idx]
            else:
                print("No bbox found for object ", obj)
        
        return res

def visualize_instance_json(image_dir, json_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("makedirs:", save_dir)

    coco = COCO(json_path)
    image_ids = list(coco.imgToAnns.keys())
    annotation_transform = COCOAnnotationTransform() # 解析annodation_dict，返回box列表
    
    for image_id in image_ids:
        ann_ids = coco.getAnnIds(imgIds=image_id) # 获取img_id所包含的所有annotation_id列表
        target = coco.loadAnns(ann_ids) # 加载ann_ids指定的所有annotation dict，返回列表
        
        ### 读取该image
        file_name = coco.loadImgs(image_id)[0]['file_name'] # 获取图像文件名
        path = os.path.join(image_dir, file_name) # 获取图像路径
        assert os.path.exists(path), 'Image path does not exist: {}'.format(path)
        image = cv2.imread(path)
        height, width, _ = image.shape
        #print("image", height, width) # image: 1920 1088

        ### 解析该image的所有instance mask, [num_objects, height, width]
        # 注意：看annToMask源码，应该用RLE的方式来表达甜甜圈圆
        masks = [coco.annToMask(obj).reshape(-1) for obj in target] # annToMask: Convert segmentation in an annotation to binary mask，值为0/1
        masks = np.vstack(masks) # (m, height*width)
        masks = masks.reshape(-1, height, width) # (m, height, width)
        #print("mask:", masks.shape, np.unique(masks)) # mask: (7, 1920, 1088) [0 1]
        
        ### 解析该image的所有box, [[xmin, ymin, xmax, ymax, class_idx], ...], xmin, ymin, xmax, ymax is normalized(0~1)
        target = annotation_transform(target, width, height) # 解析list(annodation_dict)，返回box列表
        #print("target:", len(target), target[0]) # target: 7 [0.39981617647058826, 0.3541666666666667, 0.6102941176470589, 0.4546875, 0]
        
        ### 绘制box 
        image_visual = image.copy()
        for anno in target:
            box = anno[0:-1]
            class_id = anno[-1]
            x_min, y_min, x_max, y_max = box
            x_min = int(x_min * width)
            y_min = int(y_min * height)
            x_max = int(x_max * width)
            y_max = int(y_max * height)
            image_visual = cv2.rectangle(image_visual, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        
        ### 绘制实例mask
        image_bg = cv2.merge([np.ones((height, width))*0, np.ones((height, width))*0, np.ones((height, width))*255])
        for mask in masks:
            h,w = mask.shape 
            assert h==height and w==width, path
            alpha = mask/1.0*0.45
            alpha = alpha[:,:,np.newaxis]
            image_synthetic = image_bg * alpha + (1-alpha) * image_visual 
            image_visual = image_synthetic
        
        ### 保存图像
        save_path = os.path.join(save_dir, file_name)
        cv2.imwrite(save_path, image_visual)
        print("save:", save_path)

if __name__ == "__main__":
    #image_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/生成数据/0_image_synthetic_rle_20210120"
    #json_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/生成数据/coco_instance_rle_20210120.json"
    #save_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/生成数据/0_image_synthetic_rle_20210120_visual"
    # image_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0/20210108_1260_image_train"
    # json_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0/coco_instance_20210108_1260_image_train.json"
    # save_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0/20210108_1260_image_train_visual"
    #image_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0/20210108_1260_image_train_20210127"
    #json_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0/coco_instance_20210108_1260_image_train_20210127.json"
    #save_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0/20210108_1260_image_train_20210127_visual"
    #image_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210122_504_image_test"
    #json_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/coco_instance_20210122_504_image_test.json"
    #save_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210122_504_image_test_visual"
    # image_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0/20210108_1260_train_thickness8"
    # json_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0/20210108_1260_train_thickness8.json"
    # save_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0/20210108_1260_train_thickness8_visual"
    # image_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210122_504_test_thickness8"
    # json_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210122_504_test_thickness8.json"
    # save_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210122_504_test_thickness8_visual"
    image_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0/20210205_thickness7_train"
    json_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0/20210205_thickness7_train.json"
    save_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0/20210205_thickness7_train_visual"
    # image_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210205_thickness7_test"
    # json_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210205_thickness7_test.json"
    # save_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210205_thickness7_test_visual"
    visualize_instance_json(image_dir, json_path, save_dir)
    
