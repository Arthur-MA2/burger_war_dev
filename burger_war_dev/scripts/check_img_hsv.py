#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import rospy
import os

class CheckCamImgHSV:
    width_img = 640
    height_img = 480
    ch_img = 3
    shape_img = height_img, width_img,ch_img
    h_thrshld_uppr = 70+20 #120#75+15 #90 #120
    h_thrshld_lwr  = 70-20 #30#75-15 #60 #42
    s_thrshld_uppr = 150#255 #100+30 #255
    s_thrshld_lwr  = 30 #70 #100
    v_thrshld_uppr = 200#255 #255 #120 + 30 #20
    v_thrshld_lwr  = 100 #0 #120 - 30 #20
    hight_bbox_thrshld_lwr = height_img * 0.25 #0.15 #0.2

    notfound = 404
    notfound_flg = False
    cam_img = np.zeros(shape_img)
    green_img = np.zeros(shape_img)
    rect_img = np.zeros(shape_img)

    def CheckandLookatGreenCircle(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        mask = np.zeros(h.shape, dtype=np.uint8)
        mask[(self.h_thrshld_lwr < h) & (h < self.h_thrshld_uppr) & (self.s_thrshld_lwr < s) & (s < self.s_thrshld_uppr) & (self.v_thrshld_lwr < v) & (v < self.v_thrshld_uppr)] = 255

        print("H_min" + str(min(h[(self.h_thrshld_lwr < h) & (h < self.h_thrshld_uppr) & (self.s_thrshld_lwr < s) & (s < self.s_thrshld_uppr) & (self.v_thrshld_lwr < v) & (v < self.v_thrshld_uppr)])))
        print("H_max" + str(max(h[(self.h_thrshld_lwr < h) & (h < self.h_thrshld_uppr) & (self.s_thrshld_lwr < s) & (s < self.s_thrshld_uppr) & (self.v_thrshld_lwr < v) & (v < self.v_thrshld_uppr)])))
        print("H_ave" + str(np.average(h[(self.h_thrshld_lwr < h) & (h < self.h_thrshld_uppr) & (self.s_thrshld_lwr < s) & (s < self.s_thrshld_uppr) & (self.v_thrshld_lwr < v) & (v < self.v_thrshld_uppr)])))
        print("S_min" + str(min(s[(self.h_thrshld_lwr < h) & (h < self.h_thrshld_uppr) & (self.s_thrshld_lwr < s) & (s < self.s_thrshld_uppr) & (self.v_thrshld_lwr < v) & (v < self.v_thrshld_uppr)])))
        print("S_max" + str(max(s[(self.h_thrshld_lwr < h) & (h < self.h_thrshld_uppr) & (self.s_thrshld_lwr < s) & (s < self.s_thrshld_uppr) & (self.v_thrshld_lwr < v) & (v < self.v_thrshld_uppr)])))
        print("S_ave" + str(np.average(s[(self.h_thrshld_lwr < h) & (h < self.h_thrshld_uppr) & (self.s_thrshld_lwr < s) & (s < self.s_thrshld_uppr) & (self.v_thrshld_lwr < v) & (v < self.v_thrshld_uppr)])))
        print("V_min" + str(min(v[(self.h_thrshld_lwr < h) & (h < self.h_thrshld_uppr) & (self.s_thrshld_lwr < s) & (s < self.s_thrshld_uppr) & (self.v_thrshld_lwr < v) & (v < self.v_thrshld_uppr)])))
        print("V_max" + str(max(v[(self.h_thrshld_lwr < h) & (h < self.h_thrshld_uppr) & (self.s_thrshld_lwr < s) & (s < self.s_thrshld_uppr) & (self.v_thrshld_lwr < v) & (v < self.v_thrshld_uppr)])))
        print("V_ave" + str(np.average(v[(self.h_thrshld_lwr < h) & (h < self.h_thrshld_uppr) & (self.s_thrshld_lwr < s) & (s < self.s_thrshld_uppr) & (self.v_thrshld_lwr < v) & (v < self.v_thrshld_uppr)])))

        self.green_img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circles = []
        if len(contours) > 0:
            maxCont=contours[0]
            for c in contours:
                if len(maxCont)<len(c):
                    maxCont = c
            x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(maxCont)

            if h_bbox > self.hight_bbox_thrshld_lwr:
                self.notfound_flg = False
                self.rect_img = cv2.rectangle(image,(x_bbox, y_bbox),(x_bbox+w_bbox,y_bbox+h_bbox),(0,255,0),2)
                return (x_bbox + w_bbox / 2.0) - self.width_img / 2.0
            else:
                if not self.notfound_flg:
                    self.notfound_flg = True
                    self.rect_img = np.zeros(self.shape_img)
                return self.notfound
        else:
            if not self.notfound_flg:
                self.notfound_flg = True
                self.rect_img = np.zeros(self.shape_img)
            return self.notfound


if __name__ == '__main__':
    cam_img = cv2.imread("catkin_ws/src/burger_war_dev/burger_war_dev/sample_cam_img/image6.png")

    rospy.init_node('checkCamImgHSV', anonymous=True)
    checkCamImgHSV = CheckCamImgHSV()
    cv2.namedWindow("Camera Image")

    print(os.getcwd())
    print(cam_img.shape)

    checkCamImgHSV.CheckandLookatGreenCircle(cam_img)

    r = rospy.Rate(15)
    while not rospy.is_shutdown():
        cv2.imshow("Camera Image", checkCamImgHSV.green_img)#rect_img)
        cv2.waitKey(1)
        r.sleep()
