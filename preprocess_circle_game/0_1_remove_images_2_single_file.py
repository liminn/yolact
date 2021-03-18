import os
import glob
import shutil

image_path_list = glob.glob("/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210130_446_crop/*/*/*.jpg")
new_save_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/test_0/20210130_446_crop_single" 
if not os.path.exists(new_save_dir):
    os.makedirs(new_save_dir)

for path in image_path_list:
    base_name = os.path.split(path)[1]
    new_path = os.path.join(new_save_dir, base_name)
    shutil.copy(path, new_path)
