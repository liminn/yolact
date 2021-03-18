
import os

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
    image_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/TEXT_DET/train/长单词连字符_det_20201003/origin_data"
    image_list = get_all_image_path(image_dir)
    for image_path in image_list:
        print(image_path)
        new_image_path = os.path.splitext(image_path)[0] + ".jpg"
        print(new_image_path)
        os.rename(image_path,new_image_path) 