import json
import os
import cv2
import json
import glob
import numpy as np

# temp：将json文件的bbox信息取出，存在txt文件中
# 获取目录下所有图片的路径
def get_all_image_path(dir):
    image_paths = []
    for path,d,filelist in os.walk(dir):
        for filename in filelist:
            if(filename.endswith("jpg") or filename.endswith("jpeg") or filename.endswith("png") or filename.endswith("JPG") or filename.endswith("JPEG") or filename.endswith("PNG")):
                image_paths.append((os.path.join(path, filename)))
    return image_paths

if __name__ == "__main__":
    # 待遍历的文件夹路径
    ### text
    dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/TEXT_DET/valid/en-text-detection"
    image_paths = get_all_image_path(dir)
    for image_path in image_paths:
        postfix = os.path.splitext(image_path)[-1]
        json_path = image_path.split(postfix)[0]+".json"
        txt_path = image_path.split(postfix)[0]+".txt"
        ignore_num = 0
        if os.path.isfile(json_path):
            file = open(txt_path,'w')
            with open(json_path) as f:
                pop_data = json.load(f)
                for dict_item in pop_data["shapes"]:
                    #print(dict_item["points"])
                    #print("==")
                    line = ""
                    for i in range(len(dict_item["points"])):
                        item = dict_item["points"][i]
                        line += str(item[0]) + ',' + str(item[1]) + ','
                    #print(line)
                    #print("==")
                    file.write(line + '\n')                
                file.close()
        else:
            print("no json path: {}".format(image_path))
        



       