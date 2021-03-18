"""check line label, and get line mask"""
import os
import cv2
import glob
import json
import math
import random
import numpy as np 
import tensorflow.compat.v1 as tf

model_path = "/home/dell/zhanglimin/code/page_det/saved_model"
with tf.Session(graph=tf.Graph()) as sess:
    # Load saved model
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores', 'detection_classes'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
    # image_tensor = tf.get_default_graph().get_tensor_by_name('encoded_image_string_tensor:0')
    # Input image is uint8 (RGB)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    ### define input path
    label_json_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0/circle.json"
    query_image_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0"
    json_save_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0/circle_new.json"

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
            print(image_path)
            
            ### Read image and crop image
            image_ori = cv2.imread(image_path)
            height, width, _ = image_ori.shape
            image = image_ori[:, :, [2, 1, 0]]
            image = np.expand_dims(image, axis=0)
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})
            # Scores: probalities
            scores = output_dict['detection_scores'][0]
            # Boxes: rectangles
            boxes = output_dict['detection_boxes']
            for i, score in enumerate(scores[0:1]):
                if score > 0.1:
                    detection_box = boxes[0, i]
                    #print(detection_box)
                    xmin = int(detection_box[1] * width)
                    ymin = int(detection_box[0] * height)
                    xmax = int(detection_box[3] * width)
                    ymax = int(detection_box[2] * height)

            point_dict_list = item["label"]["smaple-class"]
            for j in range(len(point_dict_list)):
                point_item = point_dict_list[j]["shape"]["geometry"]
                point_item_new = point_item.copy()
                #print(point_item)
                points_list_temp = []
                for point in point_item_new:
                    point["x"] = point["x"] - xmin
                    point["y"] = point["y"] - ymin
                point_dict_list[j]["shape"]["geometry"] = point_item_new #  更新point_item
            json_dict_list[i]["label"]["smaple-class"]= point_dict_list

    with open(json_save_path,"w") as f:
        #print(total_dict)
        json.dump(json_dict_list,f)
        print("dump:", json_save_path)
        
        



            

            

#if __name__ == "__main__":
   
    ### define save path
    #image_save_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0/20210108_1260_image_train"
    #merge_save_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/train_0/20210108_1260_image_train_visual"
    
    ### run
    #label_to_mask(label_json_path, query_image_dir, image_save_dir, merge_save_dir)
