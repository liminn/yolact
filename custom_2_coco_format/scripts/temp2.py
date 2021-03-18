import json
import os
import cv2
import json
import glob
import numpy as np

# temp：对相应的jpg/txt.json文件重命名
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
    dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/TEXT_DET"
    image_paths = get_all_image_path(dir)
    i = 0
    for image_path in image_paths:
        postfix = os.path.splitext(image_path)[-1]
        txt_path = image_path.split(postfix)[0]+".txt"
        dirname = os.path.dirname(image_path)
        #print("===")
        #print(image_path)
        #print(txt_path)
        #print("==")
        if os.path.isfile(txt_path):
            new_image_path = os.path.join(dirname, str(i)+".jpg")
            new_text_path = os.path.join(dirname, str(i)+".txt")
            #print(new_image_path)
            #print(new_text_path)
            #print("===")
            os.rename(image_path, new_image_path)
            os.rename(txt_path, new_text_path)
            i += 1
        



       