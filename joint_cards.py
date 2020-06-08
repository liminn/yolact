import os
import math
import numpy as np

def min_rectangle(xs, ys):
    xmin = np.min(xs)
    ymin = np.min(ys)
    xmax = np.max(xs)
    ymax = np.max(ys)
    return [xmin, ymin, xmax, ymax]

def preprocess(bbox_info):
    bbox_locations = []
    bbox_chars = []
    location_list = []
    for i in range(len(bbox_info)):
        char = str(bbox_info[i][0])
        x1 = float(bbox_info[i][1])
        y1 = float(bbox_info[i][2])
        x2 = float(bbox_info[i][3])
        y2 = float(bbox_info[i][4])
        x3 = float(bbox_info[i][5])
        y3 = float(bbox_info[i][6])
        x4 = float(bbox_info[i][7])
        y4 = float(bbox_info[i][8])
        xs = [x1, x2, x3, x4]
        ys = [y1, y2, y3, y4]
        # 计算正外接矩形
        xmin, ymin, xmax, ymax = min_rectangle(xs, ys)
        # 从左上开始，顺时针记录四个点, 分别为p1->p2->p3->p4
        bbox_locations.append([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])
        bbox_chars.append(char)
    return bbox_locations, bbox_chars

def get_proj_points(m, b, x0, y0):
    x1 = (m * y0 + x0 - m * b) / (m * m + 1)
    y1 = (m * m * y0 + m * x0 + b) / (m * m + 1)
    return x1, y1
    
def get_proj_len(k1, b1, x3, y3, x4, y4, y1, y2):
    x_proj_0, y_proj_0 = get_proj_points(k1, b1, x3, y3)
    x_proj_1, y_proj_1 = get_proj_points(k1, b1, x4, y4)
    y_min = max(y_proj_0, y1)
    y_max = min(y_proj_1, y2)
    # 如果不相交
    if y_min > y_max:
        return 0
    x_min = (y_min - b1) / (k1 + 0.0001)
    x_max = (y_max - b1) / (k1 + 0.0001)
    dis = math.sqrt((y_max - y_min) * (y_max - y_min) + (x_max - x_min) * (x_max - x_min))
    return dis

def compute_line_iou(x1, y1, x2, y2, x3, y3, x4, y4):
    k1 = (y1 - y2) * 1.0 / (x1 - x2 + 0.00001)
    b1 = y1 - k1 * x1  
    k2 = (y3 - y4) * 1.0 / (x3 - x4 + 0.00001)
    b2 = y3 - k2 * x3
    # 计算直线长度
    l1_len = math.sqrt((y1 - y2) * (y1 - y2) + (x1 - x2) * (x1 - x2))
    l2_len = math.sqrt((y3 - y4) * (y3 - y4) + (x3 - x4) * (x3 - x4))
    # 计算点到直线的投影点
    dis1 = get_proj_len(k1, b1, x3, y3, x4, y4, y1, y2)
    dis2 = get_proj_len(k2, b2, x1, y1, x2, y2, y3, y4)
    l_len = min(l1_len, l2_len)
    iou_line = max(dis1 / (l_len + 0.0001), dis2 / (l1_len + 0.0001))
    return iou_line
    
def get_right_nearest_bbox(bboxes_x_center, bboxes_matrix, y_min_idx):
    # 左侧
    off_x = bboxes_x_center - bboxes_x_center[y_min_idx]
    points_idx = np.where(off_x > 0)[0]

    if (points_idx.__len__() == 0):
        return -1

    bboxes_matrix_left = bboxes_matrix[points_idx]
    bbox_y0 = bboxes_matrix_left[:, 1]
    bbox_y4 = bboxes_matrix_left[:, 7]

    cur_bbox_y0 = bboxes_matrix[y_min_idx, 1]
    cur_bbox_y4 = bboxes_matrix[y_min_idx, 7]

    cur_bbox_x0 = bboxes_matrix[y_min_idx, 0]
    cur_bbox_x4 = bboxes_matrix[y_min_idx, 6]

    th_dis = ((cur_bbox_y4 - cur_bbox_y0) * (cur_bbox_y4 - cur_bbox_y0) +
            (cur_bbox_x4 - cur_bbox_x0) * (cur_bbox_x4 - cur_bbox_x0))

    bbox_iou_line = np.zeros_like(bbox_y0, np.float32)

    for i in range(bboxes_matrix_left.shape[0]):
        x2 = bboxes_matrix_left[i, 0]
        y2 = bboxes_matrix_left[i, 1]
        x3 = bboxes_matrix_left[i, 6]
        y3 = bboxes_matrix_left[i, 7]

        x1 = bboxes_matrix[y_min_idx, 2]
        y1 = bboxes_matrix[y_min_idx, 3]
        x4 = bboxes_matrix[y_min_idx, 4]
        y4 = bboxes_matrix[y_min_idx, 5]

        dis1 = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        dis2 = (x4 - x3) * (x4 - x3) + (y4 - y3) * (y4 - y3)

        dis_len = min(dis1, dis2)
        iou_line = compute_line_iou(x1, y1, x4, y4, x2, y2, x3, y3)

        if iou_line > 0.3:
            bbox_iou_line[i] = dis_len  
        else:
            bbox_iou_line[i] = np.inf

    iou_idx = np.where(bbox_iou_line < 2 * th_dis)[0]

    if iou_idx.__len__() == 0:
        return -1
    else:
        xx = np.argmin(bbox_iou_line)

        return points_idx[xx]


def get_left_nearest_bbox(bboxes_x_center, bboxes_matrix, y_min_idx):
    # bbox的左侧中心点坐标和其他bbox的右侧中点坐标的距离
    off_x = bboxes_x_center - bboxes_x_center[y_min_idx]
    points_idx = np.where(off_x < 0)[0]   # 全部最左侧的卡
    if (points_idx.__len__() == 0): # y_min_idx即为最左侧的bbox
        return -1

    # 取出全部左侧点的bbox
    bboxes_matrix_left = bboxes_matrix[points_idx]
    
    # 垂直方向上 bbox存在相交,计算相交程度
    bbox_iou_line = np.zeros_like(points_idx, np.float32)
    
    for i in range(len(points_idx)):
        # 取出待对比bbox的右侧边的两点(P2和P3)
        x1 = bboxes_matrix_left[i, 2]
        y1 = bboxes_matrix_left[i, 3]

        x4 = bboxes_matrix_left[i, 4]
        y4 = bboxes_matrix_left[i, 5]
        
        # 取出y_min_idx的左侧边的两点(P1和P4)
        x2 = bboxes_matrix[y_min_idx, 0]
        y2 = bboxes_matrix[y_min_idx, 1]

        x3 = bboxes_matrix[y_min_idx, 6]
        y3 = bboxes_matrix[y_min_idx, 7]

        dis1 = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        dis2 = (x4 - x3) * (x4 - x3) + (y4 - y3) * (y4 - y3)
        dis_len = min(dis1, dis2)
    
        iou_line = compute_line_iou(x1, y1, x4, y4, x2, y2, x3, y3)
        if iou_line > 0.3:
            bbox_iou_line[i] = dis_len  # np.where(mask > 0)[0].shape[0]
        else:
            bbox_iou_line[i] = np.inf
        
    # 如果当前未标点符号可能会出错
    # 判断iou_line 是否满足要求
    cur_bbox_x2 = bboxes_matrix[y_min_idx, 2]
    cur_bbox_y2 = bboxes_matrix[y_min_idx, 3]

    cur_bbox_x3 = bboxes_matrix[y_min_idx, 4]
    cur_bbox_y3 = bboxes_matrix[y_min_idx, 5]

    th_dis = ((cur_bbox_y3 - cur_bbox_y2) * (cur_bbox_y3 - cur_bbox_y2) +
            (cur_bbox_x3 - cur_bbox_x2) * (cur_bbox_x3 - cur_bbox_x2))
    iou_idx = np.where(bbox_iou_line < 2 * th_dis)[0]

    if iou_idx.__len__() == 0:
        return -1
    else:
        xx = np.argmin(bbox_iou_line)
        return points_idx[xx]

def line_post_process(line_res,bboxes_x_center,bboxes_y_center,bboxes_matrix):
    id_remove = []
    '''1.一个点，直接返回'''
    if len(line_res)==1:
        return line_res, id_remove
    '''2.大于两个点，间隔不能超过4倍的卡片宽度'''
    if len(line_res)>=2:
        width = bboxes_matrix[line_res[0]][2]-bboxes_matrix[line_res[0]][0]
        dist_list = [] # 记录相邻两点间隔
        for i in range(len(line_res)-1):
            x_left = bboxes_x_center[line_res[i]]
            x_right = bboxes_x_center[line_res[i+1]]
            y_left = bboxes_y_center[line_res[i]]
            y_right = bboxes_y_center[line_res[i+1]]
            dist_current =  math.sqrt((y_left - y_right) * (y_left - y_right) + (x_left - x_right) * (x_left - x_right))
            dist_list.append(dist_current)
        for i in range(len(dist_list)):
            if dist_list[i]>4*width:
                j=i
                while(j==len(dist_list)):
                    id_remove.append(line_res.pop(i+1))
                    j += 1
                return line_res, id_remove
    return line_res, id_remove

def compute_all_lines(bbox_locations):
    if len(bbox_locations) == 0:
        return []
    bboxes_matrix = np.array(bbox_locations)  # [n,8]
    
    '''stp 1 提取出中心点'''
    bboxes_x_min = np.min(bboxes_matrix[:, [0, 2, 4, 6]], axis=-1)      # [n,4]
    bboxes_x_max = np.max(bboxes_matrix[:, [0, 2, 4, 6]], axis=-1)      # [n,4]
    bboxes_y_min = np.min(bboxes_matrix[:, [1, 3, 5, 7]], axis=-1)      # [n,4]
    bboxes_y_max = np.max(bboxes_matrix[:, [1, 3, 5, 7]], axis=-1)      # [n,4]
    bboxes_x_center = bboxes_x_min + (bboxes_x_max - bboxes_x_min) / 2  # [n,1]
    bboxes_y_center = bboxes_y_min + (bboxes_y_max - bboxes_y_min) / 2  # [n,1]
    bboxes_label = np.zeros_like(bboxes_y_center)                       # [n,1]

    '''stp 2 循环遍历是否还存在行'''
    res = []
    while (1):
        no_line_idx = np.where(bboxes_label == 0)[0]  # 找到未分类的bbox
        if no_line_idx.__len__() == 0:
            return res

        line_idx = []  # 用于存放新找到的行对应的bbox的id
        '''stp 2.1 提取出这些未分类的bbox的信息'''
        bboxes_x_center_in = bboxes_x_center[no_line_idx]
        bboxes_y_center_in = bboxes_y_center[no_line_idx]
        bboxes_matrix_in = bboxes_matrix[no_line_idx]

        '''stp 2.2 从这些未分类的bbox中找到最上方的bbox'''
        y_min_idx = np.argmin(bboxes_y_center_in)  # 计算最上方bbox
        bboxes_label[no_line_idx[y_min_idx]] = 1  # 对当前的bbox对应的label=1
        line_idx.append(no_line_idx[y_min_idx])  # line_idx存放该次聚类的所有ID索引
        
        '''stp 2.3 find left'''
        y_min_idx_in = y_min_idx
        while (1):
            '''找到最左侧的bbox'''
            y_min_idx_left = get_left_nearest_bbox(bboxes_x_center_in, bboxes_matrix_in, y_min_idx_in)
            if y_min_idx_left != -1:
                bboxes_label[no_line_idx[y_min_idx_left]] = 1
                y_min_idx_in = y_min_idx_left
                line_idx.append(no_line_idx[y_min_idx_in])
            else:
                break

        y_min_idx_in = y_min_idx
        '''stp 2.4 find right'''
        while (1):
            y_min_idx_right = get_right_nearest_bbox(bboxes_x_center_in, bboxes_matrix_in, y_min_idx_in)
            if y_min_idx_right != -1:
                bboxes_label[no_line_idx[y_min_idx_right]] = 1
                y_min_idx_in = y_min_idx_right
                line_idx.append(no_line_idx[y_min_idx_in])
            else:
                break
        
        '''stp 2.5 按照bbox的中心点顺序进行排序'''
        cc = np.array(bboxes_x_center[line_idx])
        out = np.argsort(cc)
        line_res = [line_idx[i] for i in out]

        '''stp 2.6 对该次聚类结果进行后处理'''
        line_res, id_remove = line_post_process(line_res,bboxes_x_center,bboxes_y_center,bboxes_matrix)
        if len(id_remove)!=0:
            for i in range(len(id_remove)):
                bboxes_label[id_remove[i]] = 0
        
        res.append(line_res)

def get_cluster_text(bbox_chars, cluster_index):
    cluster_text = []
    for i in range(len(cluster_index)):
        text = ""
        for j in range(len(cluster_index[i])):
            text += bbox_chars[cluster_index[i][j]]
        cluster_text.append(text)
    return cluster_text
    

def get_cluster_rectangle(bbox_info, cluster_index):
    cluster_rectangle = []
    for i in range(len(cluster_index)):
        rectangle = []
        xs = []
        ys = []
        for j in range(len(cluster_index[i])):
            bbox = bbox_info[cluster_index[i][j]]
            xs.extend([int(bbox[1]),int(bbox[3]),int(bbox[5]),int(bbox[7])])
            ys.extend([int(bbox[2]),int(bbox[4]),int(bbox[6]),int(bbox[8])])
            # 计算正外接矩形
            xmin, ymin, xmax, ymax = min_rectangle(xs, ys)
            # 从左上开始，顺时针记录四个点, 分别为p1->p2->p3->p4
            rectangle = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
        cluster_rectangle.append(rectangle)    
    return cluster_rectangle

def get_results(bbox_info):
    # 1.预处理
    bbox_locations, bbox_chars = preprocess(bbox_info)
    # 2.获取bbox的聚类(分区)结果
    cluster_index = compute_all_lines(bbox_locations)
    # 3.获取分区的text文本
    cluster_text = get_cluster_text(bbox_chars, cluster_index)
    # 4.获取分区的矩形框
    cluster_rectangle = get_cluster_rectangle(bbox_info, cluster_index)
    # 5.定义整体返回结果
    cluster_dict = {"index": cluster_index, "rectangle":cluster_rectangle, "text": cluster_text }
    return cluster_dict
    