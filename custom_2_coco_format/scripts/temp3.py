import os
import json 

def parse_json_generate_txt(label_json_path, txt_save_dir):
    with open(label_json_path) as f:
        label_list = json.load(f)
        #print(len(label_list))
        #print(label_list[0])
    for i in range(len(label_list)):
        ### get all the label box of single image 
        item = label_list[i]
        image_name = item["imageInfo"]["path"]
        #print(i, image_name)
        all_box_coordinate_list = []
        box_coordinate_list = []
        #print(item["label"])
        try:
            text_box_list = item["label"]["text-box"]
        except:
            continue
        for j in range(len(text_box_list)):
            #print(text_box_list[j])
            geometry = text_box_list[j]["shape"]["geometry"]
            box_coordinate_list = []
            for point in geometry:
                box_coordinate_list.append(point["x"])
                box_coordinate_list.append(point["y"])
            all_box_coordinate_list.append(box_coordinate_list)
        ### save all_box_coordinate_list in txt
        txt_name = os.path.splitext(image_name)[0] + ".txt"
        #txt_save_path = os.path.join(txt_save_dir, txt_name)
        txt_save_path = txt_save_dir + txt_name
        # print(txt_save_dir)
        # print(txt_name)
        # print(txt_save_path)
        with open(txt_save_path, "w") as f:
            for k in range(len(all_box_coordinate_list)):
                list_temp = all_box_coordinate_list[k]
                str_line = ""
                for num in list_temp:
                    str_line += str(num) + ","
                str_line = str_line[:-1] +"\n"
                f.write(str_line)
        print(i, txt_save_path)
        
if __name__ == "__main__":
    ### 解析标注json文件
    label_json_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/TEXT_DET/train/长单词连字符_det_20201003/v2_export_281_2020-11-02_2020-11-04.json"
    txt_save_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/TEXT_DET/train/长单词连字符_det_20201003/origin_data"

    parse_json_generate_txt(label_json_path, txt_save_dir)
    