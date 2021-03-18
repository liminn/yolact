import cv2
import os
import glob
import numpy as np

if __name__ == "__main__":
    ###
    mask_list  = glob.glob("/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/生成数据/0_ps_mask/*.png")
    #mask_list = [mask_list[0]]

    ### 
    mask_save_dir = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/生成数据/0_ps_mask_crop"
    if(not os.path.exists(mask_save_dir)):
        os.makedirs(mask_save_dir)

    for j, mask_path in enumerate(mask_list):
        #print(mask_path)
        image_mask = cv2.imread(mask_path)

        #"""
        ### 切割出圆形
        mask = image_mask[:,:,-1]
        #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
        ret, binary = cv2.threshold(mask,20,255,cv2.THRESH_BINARY)  
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        #cv2.drawContours(image_mask,contours,-1,(0,0,255),3)  
        # cv2.imshow("img", img)  
        # cv2.waitKey()
        ### Find the index of the largest contour
        areas = [cv2.contourArea(c) for c in contours] 
        
        if(0 < len(areas) <= 2):
        #if(len(areas) == 1):
            # print(len(contours))
            # print(contours)
            # segmentation = []
            # for c in contours:
            #     segmentation_temp = []
            #     for point in c:
            #         point = point[0]
            #         print(point)
            #         segmentation_temp += [point[0], point[1]]
            #     segmentation.append(segmentation_temp)
            # print(segmentation)
            max_index = np.argmax(areas)
            cnt=contours[max_index]
            x,y,w,h = cv2.boundingRect(cnt)
            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            # cv2.imshow("img_2", img)  
            # cv2.waitKey()
            #new_image = np.zeros((h,w,4))
            new_image = image_mask[y-2:y+h+2, x-2:x+w+2,:]
            base_name = os.path.split(mask_path)[-1]
            image_save_path = os.path.join(mask_save_dir, base_name)
            try:
                #pass
                cv2.imwrite(image_save_path, new_image)
            except:
                continue
        #"""