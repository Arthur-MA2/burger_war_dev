#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import random

from std_msgs.msg import Int32
from geometry_msgs.msg import Twist

import tf

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib_msgs

import math
from actionlib_msgs.msg import GoalStatus
### GoalStatus: http://docs.ros.org/en/kinetic/api/actionlib_msgs/html/msg/GoalStatus.html
# 0: PENDING,   1: ACTIVE,      2: PREEMPTED,    3: SUCCEEDED,  4: ABORTED,
# 5: REJECTED,  6: PREEMPTING,  7: RECALLING,    8: RECALLED,   9: LOST


# Ref: https://hotblackrobotics.github.io/en/blog/2018/01/29/action-client-py/

#from std_msgs.msg import String
#from sensor_msgs.msg import Image
#from cv_bridge import CvBridge, CvBridgeError
#import cv2


class NaviBot:
    notfound = 404
    findFlg = False
    diff_px_x_enemy = 0

    def __init__(self):

        # velocity publisher
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

        self.lookatEnemy_sub = rospy.Subscriber('lookatEnemy', Int32, self.lookatCallback, queue_size = 1)


    def sample_setGoal(self,x,y,yaw):
        self.client.wait_for_server()

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y

        # Euler to Quartanion
        q=tf.transformations.quaternion_from_euler(0,0,yaw)
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]

        self.client.send_goal(goal)
        wait = self.client.wait_for_result()
        if not wait:
            rospy.logerr("Action server not available!")
            rospy.signal_shutdown("Action server not available!")
        else:
            return self.client.get_result()


    def sample_strategy(self):
        r = rospy.Rate(5) # change speed 5fps

        self.setGoal(-0.5,0,0)
        self.setGoal(-0.5,0,3.1415/2)

        self.setGoal(0,0.5,0)
        self.setGoal(0,0.5,3.1415)

        self.setGoal(-0.5,0,-3.1415/2)

        self.setGoal(0,-0.5,0)
        self.setGoal(0,-0.5,3.1415)

        rospy.loginfo("FINISH!!!")

    def setGoal(self,goal_pose):#x,y,yaw):
        [x, y, yaw] = goal_pose
        self.client.wait_for_server()

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y

        # Euler to Quartanion
        q=tf.transformations.quaternion_from_euler(0,0,yaw)
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]

        self.client.send_goal(goal)

        return self.client.get_state()

    def run_strategy(self):
        r = rospy.Rate(10) # change speed 10[Hz]
        #           [     x,        y,        yaw]
        #           [gzbo_y,  -gzbo_x,     ???-(gzbo_yaw-math.pi/2.0)???]
        waypoint = [[-1.3,      0.0,       0.0], \
                    [-0.894,   -0.359,    -(2.215-math.pi/2.0)], \
                    [-1.3,      0.0,       0.0], \
                    [-0.6,      0.0,       0.0], \
                    [-0.95,     0.0,       0.0], \
                    [-1.3,      0.0,       0.0], \
                    [-0.894,    0.359,     2.215-math.pi/2.0], \
                    [-0.406,    0.923,     2.069-math.pi/2.0], \
                    \
                    [ 0.0,      1.3,      -math.pi/2.0], \
                    [-0.230,    0.802,    -math.pi/2.0-0.621], \
                    [ 0.0,      1.3,      -math.pi/2.0], \
                    [ 0.0,      0.6,      -math.pi/2.0], \
                    [ 0.0,      0.95,     -math.pi/2.0], \
                    [ 0.0,      1.3,      -math.pi/2.0], \
                    [ 0.230,    0.802,    -math.pi/2.0+0.621], \
                    [ 0.923,    0.406,    -math.pi/4.0], \
                    \
                    [ 1.3,      0.0,       math.pi], \
                    [ 0.894,    0.359,     math.pi-(2.215-math.pi/2.0)], \
                    [ 1.3,      0.0,       math.pi], \
                    [ 0.6,      0.0,       math.pi], \
                    [ 0.95,     0.0,       math.pi], \
                    [ 1.3,      0.0,       math.pi], \
                    [ 0.894,   -0.359,     math.pi+(2.215-math.pi/2.0)], \
                    [ 0.406,   -0.923,     math.pi+(2.069-math.pi/2.0)], \
                    \
                    [ 0.0,     -1.3,       math.pi/2.0], \
                    [ 0.230,   -0.802,     math.pi/2.0-0.621], \
                    [ 0.0,     -1.3,       math.pi/2.0], \
                    [ 0.0,     -0.6,       math.pi/2.0], \
                    [ 0.0,     -0.95,      math.pi/2.0], \
                    [ 0.0,     -1.3,       math.pi/2.0], \
                    [-0.230,   -0.802,     math.pi/2.0+0.621], \
                    [-0.923,   -0.406,     math.pi/2.0+math.pi/4.0]
                    ]

        wp_count = 0
        wp_num = len(waypoint)

        twist = Twist()
        twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
        kp = 0.00888 * 0.5 #1 #0.2 # px to rad/sec

        self.setGoal(waypoint[wp_count])
        while not rospy.is_shutdown():
            rospy.loginfo("!!!Loop!!!")
            if self.diff_px_x_enemy != self.notfound and not self.findFlg:
                rospy.loginfo("!!!True!!!")
                self.findFlg = True
                #self.client.cancel_goal()
                self.client.cancel_all_goals()
                twist.angular.z = - self.diff_px_x_enemy * kp
                rospy.loginfo(twist.angular.z)
                self.vel_pub.publish(twist)
                #findFlg = False
            elif self.findFlg:
                twist.angular.z = 0.0
                self.vel_pub.publish(twist)
                self.findFlg = False
            else:
                if self.client.get_state() == GoalStatus.SUCCEEDED or self.client.get_state() == GoalStatus.PREEMPTED:
                    wp_count += 1
                    if wp_count >= wp_num:
                        wp_count = 0
                    self.setGoal(waypoint[wp_count])
            rospy.loginfo(self.client.get_state())

            r.sleep()
            #self.client.cancel_goal()

    def lookatCallback(self, data):
        #self.findFlg = True
        self.diff_px_x_enemy = data.data


if __name__ == '__main__':
    rospy.init_node('navirun')
    bot = NaviBot()
    bot.run_strategy()
