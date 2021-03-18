import os
import cv2
import glob
import json
import random
import numpy as np
import pycocotools
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
"""
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of 
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask, 
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs

def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)
"""
### 合成数据，并返回coco_instance_format所需信息
def synthesize_data_instance_format(image_query, image_mask, image_bg, mask_num, image_mask_total, RLE=False):

    ### 缩放image_mask
    h,w,c = image_query.shape
    if mask_num > 10:
       h_new = np.random.randint(h*0.05, h*0.1)
    else:
       h_new = np.random.randint(h*0.05, h*0.2)
    ratio = h/h_new
    w_new = int(w/ratio)
    assert h_new<h and w_new<w
    image_mask = cv2.resize(image_mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
    #image_mask_origin = image_mask.copy()
    
    ### 腐蚀膨胀
    image_mask = np.array(image_mask).astype(np.float32)   
    #if random.uniform(0,1) <= 0.5:
    if random.uniform(0,1) <= -1: # 只膨胀
        erode_kernel_size  = np.random.randint(1, 3)
        erode_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_kernel_size, erode_kernel_size))
        image_mask = cv2.erode(image_mask, erode_kernel)
    else:
        dilate_kernel_size = np.random.randint(1, 3)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size))
        image_mask = cv2.dilate(image_mask, dilate_kernel)
    
    """
    ### 施加homography
    perspective = 0.0012
    H = np.eye(3)
    H[2, 0] = random.uniform(-perspective*0.1, perspective*0.1)  # x perspective (about y)
    H[2, 1] = random.uniform(-perspective, 0)  # y perspective (about x)
    im_out = cv2.warpPerspective(image_mask, H, (w*3,h*3))
    cv2.imwrite("im_out.png", im_out)
    """

    ### 随机指定mask粘贴位置
    image_mask_new = np.zeros(image_query.shape)
    h,w,c = image_query.shape
    h_mask,w_mask,c = image_mask.shape
    border_w = int(w*0.05)
    border_h = int(h*0.05)
    #x_start = random.randint(0, w-w_mask)
    x_start = random.randint(border_w, w-w_mask-border_w)
    #y_start = random.randint(0, h-h_mask)
    y_start = random.randint(border_h, h-h_mask-border_h)
    image_mask_new[y_start:y_start+h_mask,x_start:x_start+w_mask,:] = image_mask
    image_mask_new = image_mask_new.astype(np.uint8)

    ### 检查新粘贴的位置是否和image_mask_total有重叠
    image_mask_check = image_mask_total[y_start:y_start+h_mask,x_start:x_start+w_mask,:].copy()
    image_mask_check[image_mask_check<5] = 0
    image_mask_check[image_mask_check>=5] = 1
    mask_sum = float(np.sum(image_mask_check))
    if(mask_sum > 0): # 有重叠，跳过该次合成
        return image_query, image_mask_total, None, None, None, None
    else:
        image_mask_total[y_start:y_start+h_mask,x_start:x_start+w_mask,:] = image_mask
    
    ### 获取segmentation/area/bbox 
    mask = image_mask_new[:,:,-1].copy()
    #ret, binary = cv2.threshold(mask,5,255,cv2.THRESH_BINARY)  
    ret, binary = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)  
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    areas = [cv2.contourArea(c) for c in contours] 
    segmentation = None
    area = None
    bbox = None
    if(0 < len(areas) <= 2):
    #if(len(areas) == 1):
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
        mask_temp = image_mask_new[:,:,-1].copy()
        mask_temp[mask_temp<5] = 0
        mask_temp[mask_temp>=5] = 1
        area = float(np.sum(mask_temp))
    ### 获取mask的RLE表示
    rle = None
    if RLE:
        #rle = rle_encode(binary).tolist()
        #rle = rle_encode(binary)
        #rle = rle_to_string(rle)
        #rle = [rle_to_string(rle)]

        bin_mask = np.asfortranarray(image_mask_new[:,:,0]) 
        bin_mask[bin_mask<5] = 0
        bin_mask[bin_mask>=5] = 1
        #print(binmask)
        #rle = maskUtils.encode(bin_mask)
        #rle = COCO.encodeMask(bin_mask)
        ### ref: https://github.com/dbolya/yolact/issues/544
        rle = pycocotools.mask.encode(np.asfortranarray(bin_mask.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')  # json.dump doesn't like bytes strings
        #rle = str(rle)
        #print(rle)

    ### 合成图像，获取image_synthetic
    image_visual = image_query.copy()
    alpha = image_mask_new/255.
    """注意：合成，用原始image_mask来合成。但是，label中的mask已经进行了dilate!
    h_mask,w_mask,c = image_mask_origin.shape
    image_mask_temp = np.zeros(image_query.shape)
    image_mask_temp[y_start:y_start+h_mask,x_start:x_start+w_mask,:] = image_mask_origin
    image_mask_temp = image_mask_temp.astype(np.uint8)
    alpha = image_mask_temp/255.
    alpha *= np.random.uniform(0.3,1.0)
    """
    image_synthetic = image_bg * alpha + (1-alpha) * image_visual 
    
    return image_synthetic, image_mask_total, segmentation, area, bbox, rle
    
if __name__ == "__main__":
    ### define images and mask
    image_list = glob.glob("/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/生成数据/0_query_crop/*/*.jpg")
    image_list = image_list*5
    mask_list = glob.glob("/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/生成数据/0_ps_mask_crop/*.png")
    
    ### define image/json save dir
    image_save_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/生成数据/0_image_synthetic_rle_20210120"
    if(not os.path.exists(image_save_dir)):
        os.makedirs(image_save_dir)
    json_save_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/生成数据/coco_instance_rle_20210120.json"
    
    ### 初始化coco dict
    image_id = 0
    instance_id = 0
    total_dict = coco_instance_init()
    RLE = True #True
    #print(total_dict)

    ### 遍历所有空白query图像
    for i, image_path in enumerate(image_list):
        image_query = cv2.imread(image_path)
        ### 先缩放图像，长边为550
        h,w,c = image_query.shape
        max_size = 550
        if h > w:
            h_new = max_size
            w_new = w * h_new/h
        else:
            w_new = max_size
            h_new = h * w_new/w
        image_query = cv2.resize(image_query, (int(w_new), int(h_new)))
        ### 维持一个总体mask，确保生成的mask不重叠
        h,w,c = image_query.shape
        image_mask_total = np.zeros((h,w,c)) 
        ### 每张query，选取n个mask来进行合成
        mask_num = random.randint(5, 20)
        #mask_num = 5 #random.randint(3, 15)
        mask_path_list_selected = random.sample(mask_list, mask_num)
        
        ### 遍历所有选中的mask
        get_instance = False
        for j, mask_path in enumerate(mask_path_list_selected):
            image_mask = cv2.imread(mask_path)
            image_bg = cv2.merge([np.ones((h, w))*0, np.ones((h, w))*0, np.ones((h, w))*0])
            
            image_synthetic, image_mask_total_new, segmentation, area, bbox, rle = synthesize_data_instance_format(image_query, image_mask, image_bg, mask_num, image_mask_total, RLE)
            #print(list(rle))
            if RLE:
                if bbox is None or rle is None:
                    print("bbox/rle is None, continue")
                    continue
            else:
                if segmentation is None or bbox is None or area is None:
                    print("segmentation/bbox/area is None, continue")
                    continue
            get_instance = True
            image_query = image_synthetic # 更新image_query
            image_mask_total = image_mask_total_new # 更新image_mask_total
            ### 获取描述instance的dict，并添加进total_dict
            instance_dict= get_coco_instance_dict(image_id, instance_id, segmentation, area, bbox, rle, h, w, RLE)
            assert 1<=len(instance_dict["segmentation"])<=2
            total_dict["annotations"].append(instance_dict)
            instance_id += 1 # 更新instance_id
        
        if get_instance:
            ### 保存合成后的query image
            base_name = os.path.split(image_path)[-1]
            postfix = os.path.splitext(image_path)[-1]
            prefix = os.path.splitext(base_name)[0]
            #image_save_path = os.path.join(image_save_dir, str(image_id)+"_"+prefix+".jpg")
            image_save_path = os.path.join(image_save_dir, prefix+"_" +str(image_id)+".jpg")
            cv2.imwrite(image_save_path, image_query)
            print("{}/{}, save synthesitic image:{}".format(image_id, len(image_list), image_save_path))

            ###获取描述image(合成之后)的dict，并添加进total_dict
            image_dict= get_coco_image_dict(image_save_path,image_id)
            total_dict["images"].append(image_dict)
            image_id += 1 # 更新image_id

    ### 保存total_dict至json文件
    #json_save_path = os.path.join(json_save_dir, json_name)
    with open(json_save_path,"w") as f:
        #print(total_dict)
        json.dump(total_dict,f)
        print("dump:", json_save_path)
    