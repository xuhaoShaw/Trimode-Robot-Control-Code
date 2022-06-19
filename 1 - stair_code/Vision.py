import numpy as np
import cv2 as cv

'''通用参数'''
DETECT_THRESHOLD = 2000 # 识别距离
GROUND = 300 # 相机高度坐标, y轴向下
GROUND_ERROR = 20 # 地面误差

'''上楼检测参数'''
ABOVE_THRESHOLD = 10000 # 高于地面点数，判定上楼/障碍物
UP_LENGTH_THRESHOLD = 100 # 直线长度阈值
UP_DISTANCE_THRESHOLD = 3 # 直线外点阈值
UP_CANNY_TH1 = 3000 # 梯度高于该值且与轮廓点相连，视作轮廓
UP_CANNY_TH2 = 5000 # 梯度高于该值视作轮廓

'''下楼检测参数'''
BELOW_THRESHOLD = 10000  # 低于地面点数，判定下楼
DOWN_LENGTH_THRESHOLD = 200  # 直线长度阈值
DOWN_DISTANCE_THRESHOLD = 5  # 直线外点阈值
DOWN_CANNY_TH1 = 3000  # 梯度高于该值且与轮廓点相连，视作轮廓
DOWN_CANNY_TH2 = 5000  # 梯度高于该值，视作轮廓

class Vision:
    def __init__(self):
        self.w = 640
        self.h = 480
        self.fx = 611.36
        self.fy = 611.36
        self.cx = 320.917 # cx
        self.cy = 251.071 # cy

        self.K = np.eye(3)
        self.K[0, 0] = self.fx # fx
        self.K[1, 1] = self.fy # fy
        self.K[0, 2] = self.cx # cx
        self.K[1, 2] = self.cy # cy

        self.fast_upedge = cv.ximgproc.createFastLineDetector( #检测直线
                         _length_threshold=UP_LENGTH_THRESHOLD,
                         _distance_threshold=UP_DISTANCE_THRESHOLD,
                         _canny_th1=UP_CANNY_TH1,
                         _canny_th2=UP_CANNY_TH2,
                         _canny_aperture_size=5,
                         _do_merge=True)
        self.fast_downedge = cv.ximgproc.createFastLineDetector(
                             _length_threshold=DOWN_LENGTH_THRESHOLD,
                             _distance_threshold=DOWN_DISTANCE_THRESHOLD,
                             _canny_aperture_size=0,
                             _do_merge=True)

    # 处理一帧图片
    def process_frame(self, color, depth):
        self.depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth, alpha=0.03), cv.COLORMAP_JET)
        _, threshed_depth = cv.threshold(depth, DETECT_THRESHOLD, 1, cv.THRESH_TOZERO_INV) #二值化处理
        _, mask = cv.threshold(depth, DETECT_THRESHOLD, 1, cv.THRESH_BINARY_INV)

        depth_3d = cv.rgbd.depthTo3d(threshed_depth, self.K) * 1000
        y = depth_3d[:,:,1]
        case, case_mask = self.judge_ground(y) # 判断前方路况
        if case == 'plane':
            return 'ahead', None
        elif case == 'above':
            flag, edge = self.judge_upedge(color, threshed_depth, case_mask)
            if flag: # 上楼
                edge = edge.astype(np.int32)
                x1 = depth_3d[edge[1], edge[0], 0]
                z1 = depth_3d[edge[1], edge[0], 2]
                x2 = depth_3d[edge[3], edge[2], 0]
                z2 = depth_3d[edge[3], edge[2], 2]
                line = [x1, z1, x2, z2] # 接口，直线的端点坐标
                return 'upstair', line
            else: # 避障
                x1 = (edge[0]-self.cx)/(edge[1]-self.cy) * GROUND
                z1 = self.fy/(edge[1]-self.cy) * GROUND
                x2 = (edge[2]-self.cx)/(edge[3]-self.cy) * GROUND
                z2 = self.fy/(edge[3]-self.cy) * GROUND
                line = [x1, z1, x2, z2] # 接口，障碍物包围框前边缘端点坐标
                return 'detour', line
        elif case == 'below':
            flag, edge = self.judge_downedge(y, case_mask)
            if flag: # 下楼
                x1 = (edge[0]-self.cx)/(edge[1]-self.cy) * GROUND
                z1 = self.fy/(edge[1]-self.cy) * GROUND
                x2 = (edge[2]-self.cx)/(edge[3]-self.cy) * GROUND
                z2 = self.fy/(edge[3]-self.cy) * GROUND
                line = [x1, z1, x2, z2] # 接口，直线的端点坐标
                return 'downstair', line
            else: # 未知
                return 'stop', None

    # 检测地面点数量
    def judge_ground(self, y):
        # y = depth_3d[:,:,1]*1000
        _, above_mask = cv.threshold(y, GROUND-GROUND_ERROR, 1, cv.THRESH_BINARY_INV) # 高出地面一定距离
        _, below_mask = cv.threshold(y, GROUND+GROUND_ERROR, 1, cv.THRESH_BINARY) # 低于地面一定距离
        cv.imshow('above', above_mask)
        # cv.imshow('below', below_mask)
        cv.waitKey()
        a = np.sum(above_mask)
        b = np.sum(below_mask)

        if b > BELOW_THRESHOLD:
            return 'below', below_mask
        elif a > ABOVE_THRESHOLD:
            return 'above', above_mask
        else:
            return 'plane', None

    # 识别上楼边缘
    def judge_upedge(self, color, threshed_depth, above_mask):
        edgeline = self.fast_upedge.detect((threshed_depth*above_mask).astype(np.uint8))
        if edgeline is None: # 未识别到上楼边缘，判定有障碍物
            masked_depth = cv.boundingRect(above_mask) # 获取障碍物轮廓
            return False, masked_depth[[0,3,2,3]]
        else: # 识别到上楼边缘
            v = edgeline[:,:,1] + edgeline[:,:,3]
            lineid = np.argmax(v)

            # line = self.fast_upedge.drawSegments(self.depth_colormap, edgeline)
            line = self.fast_upedge.drawSegments(color, edgeline[3:4])
            cv.imshow('line', line)
            cv.waitKey()
            return True, edgeline[lineid].reshape(4)

    # 识别下楼边缘
    def judge_downedge(self, y, below_mask):
        ground_y = y * (1-below_mask)
        # cv.imshow('y', y)
        # cv.imshow('g', ground_y)
        # cv.waitKey()
        edge = cv.Canny(ground_y.astype(np.uint8), DOWN_CANNY_TH1, DOWN_CANNY_TH2, apertureSize=5)
        edgeline = self.fast_downedge.detect(edge)
        if edgeline is None : # 未识别到下楼边缘，未知状态
            return False, None
        else: # 识别到下楼边缘
            v = edgeline[:,:,1] + edgeline[:,:,3]
            lineid = np.argmax(v)

            line = self.fast_downedge.drawSegments(self.depth_colormap, edgeline)
            cv.imshow('line', line)
            cv.waitKey()
            return True, edgeline[lineid].reshape(4)
