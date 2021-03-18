import cv2
import random
import numpy as np

if __name__ == "__main__":
    path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/画圈游戏/生成数据/0_ps_mask/0.png"
    image_mask = cv2.imread(path)
    #image_mask = cv2.imread("1.png")
    h,w,c = image_mask.shape
    cv2.imwrite("mask.png", image_mask)
    
    """
    ### 施加homography
    perspective = 0.0012
    H = np.eye(3)
    H[2, 0] = random.uniform(-perspective*0.1, perspective*0.1)  # x perspective (about y)
    H[2, 1] = random.uniform(-perspective, 0)  # y perspective (about x)
    im_out = cv2.warpPerspective(image_mask, H, (w*3,h*3))
    cv2.imwrite("im_out.png", im_out)
    """

    """
    ### 腐蚀膨胀
    image_mask = np.array(image_mask).astype(np.float32)    
    h, w, c = image_mask.shape
    erode_kernel_size  = np.random.randint(1, 4)
    dilate_kernel_size = np.random.randint(1, 4)
    
    erode_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_kernel_size, erode_kernel_size))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size))
    image_mask_eroded = cv2.erode(image_mask, erode_kernel)
    image_mask_dilated = cv2.dilate(image_mask, dilate_kernel)
    cv2.imwrite("image_mask_eroded.png", image_mask_eroded)
    cv2.imwrite("image_mask_dilated.png", image_mask_dilated)
    """

    """
    ###旋转
    center = (w // 2, h // 2)
    degree = np.random.randint(-45, 45)
    M = cv2.getRotationMatrix2D(center, degree, 1)
    rotated = cv2.warpAffine(image_mask, M, (w, h))
    cv2.imwrite("rotated.jpg", rotated)
    """
    
    """
    ###缩放
    h_new = np.random.randint(100, 300)
    ratio = h/h_new
    w_new = int(w/ratio)
    image_resize = cv2.resize(image_mask, (w_new, h_new))
    cv2.imwrite("resize.jpg", image_resize)
    """

    """
    ###合成
    image_query = cv2.imread("query.jpg")
    image_visual = image_query.copy()
    alpha = image_mask/255.
    alpha = cv2.resize(alpha, (image_query.shape[1],image_query.shape[0]))
    image_bg = cv2.merge([np.ones((image_query.shape[0], image_query.shape[1]))*0, np.ones((image_query.shape[0], image_query.shape[1]))*0, np.ones((image_query.shape[0], image_query.shape[1]))*255])
    image_merge = image_bg * alpha + (1-alpha) * image_visual 
    cv2.imwrite("merge.jpg", image_merge)
    """

    ###切割出圆形
    mask = image_mask[:,:,-1]
    print(mask.dtype)
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    ret, binary = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)  
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    # cv2.drawContours(img,contours,-1,(0,0,255),3)  
    # cv2.imshow("img", img)  
    # cv2.waitKey()
    # Find the index of the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    # cv2.imshow("img_2", img)  
    # cv2.waitKey()
    #new_image = np.zeros((h,w,4))
    new_image = image_mask[y-5:y+h+5, x-5:x+w+5,:]
    cv2.imwrite("new_mask.jpg", new_image)

    ###


    

