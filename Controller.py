import rospy
from sensor_msgs.msg import JointState

from std_msgs.msg import UInt8MultiArray

from std_msgs.msg import Bool
import dvrk
import numpy as np
from os.path import join
import torch
from Net import *
from loadModel import get_model, load_model
import time

from AnalyticalModel import *


class Controller():
    safe_upper_torque_limit_arr = np.array([0.2, 0.8, 0.6, 0.2, 0.2, 0.2, 0])
    safe_lower_torque_limit_arr = np.array([-0.2, -0.1, 0, -0.3, -0.1, -0.1, 0])
    db_vel_arr = np.array([0.02, 0.02, 0.02, 0.01, 0.008, 0.008, 0.01])
    sat_vel_arr = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    fric_comp_ratio_arr = np.array([0.7, 0.01, 0.5, 0.4, 0.2, 0.2, 1])
    GC_init_pos_arr = np.radians(np.array([0, 0, 0, 0, 90, 0, 0]))
    safe_vel_limit = np.array([6,6,6,6,6,6,100])
    D = 6
    device = 'cpu'
    count = 0
    model = None
    isExceedSafeVel = False
    isOutputGCC = False


    def __init__(self, MTM_ARM):

        # define ros node
        rospy.init_node(MTM_ARM + 'controller', anonymous=True)

        self.MTM_ARM = MTM_ARM

        # define topics
        self.pub_tor_topic = '/dvrk/' + MTM_ARM + '/set_effort_joint'
        self.sub_pos_topic = '/dvrk/' + MTM_ARM + '/state_joint_current'
        self.pub_isFloatMode_topic = '/dvrk/' + MTM_ARM + '/set_floating_mode'
        self.pub_isDefaultGCC_topic = '/dvrk/' + MTM_ARM + '/set_gravity_compensation'

        # define publisher
        self.pub_tor = rospy.Publisher(self.pub_tor_topic, JointState, queue_size=15)
        self.pub_isFloatMode = rospy.Publisher(self.pub_isFloatMode_topic, UInt8MultiArray, queue_size=15)
        self.pub_isDefaultGCC = rospy.Publisher(self.pub_isDefaultGCC_topic, Bool, queue_size=15)

        self.set_default_GCC_mode(False)
        self.mtm_arm = dvrk.mtm(MTM_ARM)

        time.sleep(1)

        self.sub_pos = rospy.Subscriber(self.sub_pos_topic, JointState, self.sub_pos_cb_with_gcc)


    def load_gcc_model(self, model_type, load_model_path=None, use_net=None, train_type=None):
        if model_type == 'analytical_model':
            self.model =  MTM_MLSE4POL()
        elif model_type == 'DFNN':

            self.model = get_model('MTM', use_net, self.D, device='cpu')
            self.model, _, _ = load_model(load_model_path, use_net + '_' + train_type,  self.model)
        else:
            raise Exception("model type is not support.")

    def start_gc(self):
        self.set_floating_mode(False)
        self.mtm_arm.move_joint(self.GC_init_pos_arr)
        self.set_isOutputGCC(True)
        self.set_floating_mode(True)
        print("GCC start")


    def stop_gc(self):
        self.set_isOutputGCC(False)
        self.mtm_arm.move_joint(self.GC_init_pos_arr)
        print("GCC stop")

    def update_isExceedSafeVel(self, vel_arr):
        abs_vel_arr = np.abs(vel_arr)
        for i in range(self.D):
            if  abs_vel_arr[i] > self.safe_vel_limit[i]:
                self.isExceedSafeVel =  True
                break
            else:
                self.isExceedSafeVel = False


    def dbs_vel(self, joint_vel, bd_vel, sat_vel, fric_comp_ratio):
        """
        Deadband segmented function
        :param bd_vel: dead band velocity
        :param sat_vel: saturated velocity
        :param fric_comp_ratio: compensate ratio
        :return: direction coeifficent 0~1
        """
        if joint_vel >= sat_vel:
            sign_vel = 0.5 + 0.5 * fric_comp_ratio;
        elif joint_vel <= -sat_vel:
            sign_vel = 0.5 - 0.5 * fric_comp_ratio;
        elif joint_vel <= bd_vel and joint_vel >= -bd_vel:
            sign_vel = 0.5
        elif joint_vel > bd_vel and joint_vel < sat_vel:
            sign_vel = 0.5 * fric_comp_ratio * (joint_vel - bd_vel) / (sat_vel - bd_vel) + 0.5
        elif joint_vel < -bd_vel and joint_vel > -sat_vel:
            sign_vel = -0.5 * fric_comp_ratio * (-joint_vel - bd_vel) / (sat_vel - bd_vel) + 0.5
        else:
            raise Exception("joint_vel is not in if range")

        return sign_vel

    # call back function for subscribe position topic applying GCC
    def sub_pos_cb_with_gcc(self, data):

        # test function collapse time
        start = time.clock()

        pos_lst = data.position[:-1]  # select 1-6 joints
        vel_lst = data.velocity[:-1]
        effort_lst = data.effort[:-1]

        pos_arr = np.array(pos_lst)
        vel_arr = np.array(vel_lst)
        effort_arr = np.array(effort_lst)

        tor_arr = self.predict(pos_arr, vel_arr)

        self.count += 1
        if (self.count == 50):
            print('predict:', tor_arr)
            print('measure:', effort_arr)
            print('error:', tor_arr - effort_arr)
            self.count = 0

        tor_arr = self.bound_tor(tor_arr)


        msg = JointState()
        output_lst = tor_arr.tolist()
        output_lst.append(0.0)

        msg.effort = output_lst

        if self.isOutputGCC:
            self.pub_tor.publish(msg)

        self.update_isExceedSafeVel(vel_arr)

        # elapsed = time.clock()
        # elapsed = elapsed - start
        # print "Time spent in (function name) is: ", elapsed

    # model predict function
    def predict(self, pos_arr, vel_arr):
        """
        :param SinCos_pos_arr: [sin(q), cos(q)]
        :param vel_arr: velocity array
        :return: tor_arr
        """

        if self.model is not None:
            pos = pos_arr.reshape(1,-1)
            SinCos_pos = np.concatenate((np.sin(pos), np.cos(pos)), axis=1)
            tor_pos = self.model.predict_NP(np.concatenate((SinCos_pos, np.ones((1, self.D))), axis=1))
            tor_neg = self.model.predict_NP(np.concatenate((SinCos_pos, np.zeros((1, self.D))), axis=1))

            sign_vel_vec = np.zeros((1, 6))
            for i in range(6):
                sign_vel_vec[0][i] = self.dbs_vel(vel_arr[i], self.db_vel_arr[i], self.sat_vel_arr[i], self.fric_comp_ratio_arr[i])

            tor = np.multiply(tor_pos, sign_vel_vec) + np.multiply(tor_neg, 1 - sign_vel_vec)
            tor_arr = tor[0]
        else:
            tor_arr = np.zeros((self.D))
        return tor_arr

    # saturate the output torques
    def bound_tor(self, tor_arr):
        tor = tor_arr
        for i in range(self.D):
            if tor[i] >= self.safe_upper_torque_limit_arr[i]:
                tor[i] = self.safe_upper_torque_limit_arr[i]
            elif tor[i] <= self.safe_lower_torque_limit_arr[i]:
                tor[i] = self.safe_lower_torque_limit_arr[i]
        return tor

    # publish topic: set_floating_mode
    def set_floating_mode(self, is_enable):
        msg = UInt8MultiArray()
        if is_enable:
            msg.data = [1, 1, 1, 1, 1, 1, 1]
        else:
            msg.data = [0, 0, 0, 0, 0, 0, 0]

        self.pub_isFloatMode.publish(msg)

    # publish topic: set_gravity_compensation
    def set_default_GCC_mode(self, is_enable):
        msg = Bool()
        if is_enable:
            msg.data = True
        else:
            msg.data = False

        self.pub_isDefaultGCC.publish(msg)

    def ros_spin(self):
        while not rospy.is_shutdown():
            pass
        self.stop_gc()


    def set_isOutputGCC(self, isOutputGCC):
        self.isOutputGCC = isOutputGCC





# test controller function
MTM_ARM = 'MTMR'
use_net = 'ReLU_Dual_UDirection'
load_model_path = join("data", "MTMR_28002", "real", "uniform", "N4", 'D6_SinCosInput', "dual", "result", "model")
train_type = 'PKD'
model_type = 'DFNN'



controller = Controller(MTM_ARM)
controller.load_gcc_model(model_type, load_model_path=load_model_path, use_net=use_net, train_type=train_type)
controller.start_gc()
controller.ros_spin()

