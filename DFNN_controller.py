import rospy
from sensor_msgs.msg import JointState
#import pdb
#pdb.set_trace()
from evaluateTool import predict
import dvrk
import numpy as np
from os.path import join
import torch
from Net import *
from loadModel import get_model, load_model
import time

from AnalyticalModel import *

# Deadband segmented friction
# sign_vel = 0~1 function
def dbs_vel(joint_vel, bd_vel, sat_vel, fric_comp_ratio):
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


def callback(data):
    global model_type
    global pub
    global count
    global model
    global safe_upper_torque_limit_arr
    global safe_lower_torque_limit_arr
    global db_vel_arr
    global sat_vel_arr
    global fric_comp_ratio_arr
    global GC_init_pos_arr


    start = time.clock()


    pos_lst = data.position[:-1]
    vel_lst = data.velocity[:-1]
    effort_lst = data.effort[:-1]

    pos_arr = np.array(pos_lst).reshape(1,-1)
    SinCos_pos_arr = np.concatenate((np.sin(pos_arr), np.cos(pos_arr)), axis=1)
    vel_arr = np.array(vel_lst)
    effort_arr = np.array(effort_lst)

    if model_type == 'ReLU_Dual_UDirection':
        #print(SinCos_pos_arr)
        tor_pos = model.predict_NP(np.concatenate((SinCos_pos_arr, np.ones(pos_arr.shape)), axis=1))
        tor_neg = model.predict_NP(np.concatenate((SinCos_pos_arr, np.zeros(pos_arr.shape)), axis=1))

        sign_vel_vec = np.zeros((1,6))
        for i in range(6):
            sign_vel_vec[0][i] = dbs_vel(vel_arr[i], db_vel_arr[i], sat_vel_arr[i], fric_comp_ratio_arr[i])

        tor_arr = np.multiply(tor_pos,sign_vel_vec) + np.multiply(tor_neg,1-sign_vel_vec)

        tor_arr = tor_arr[0]

        # saturate the output torques
        for i in range(6):
            if tor_arr[i] >= safe_upper_torque_limit_arr[i]:
                tor_arr[i] = safe_upper_torque_limit_arr[i]
            elif tor_arr[i] <= safe_lower_torque_limit_arr[i]:
                tor_arr[i] = safe_lower_torque_limit_arr[i]



    if (count == 50):
        print('predict:', tor_arr)
        print('measure:',effort_arr)
        print('error:',tor_arr-effort_arr)
        count = 0
    else:
        count = count+1

    msg = JointState()
    output_lst = tor_arr.tolist()
    output_lst.append(0.0)

    msg.effort = output_lst

    #pub.publish(msg)



    # elapsed = time.clock()
    # elapsed = elapsed - start
    # print "Time spent in (function name) is: ", elapsed




def loop_func(MTM_ARM, use_net, load_model_path, train_type):
    global model_type
    model_type = use_net
    global pub
    global count
    global model
    global safe_upper_torque_limit_arr
    global safe_lower_torque_limit_arr
    global db_vel_arr
    global sat_vel_arr
    global fric_comp_ratio_arr
    global GC_init_pos_arr



    safe_upper_torque_limit_arr = np.array([0.2,0.8,0.6,0.2,0.2,0.2,0])
    safe_lower_torque_limit_arr = np.array([-0.2,-0.1,0,-0.3,-0.1,-0.1,0])
    db_vel_arr = np.array([0.02,0.02,0.02,0.01,0.008,0.008,0.01])
    sat_vel_arr = np.array([0.2,0.2,0.2,0.2,0.2,0.2,0.2])
    fric_comp_ratio_arr = np.array([0.7,0.01,0.5,0.4,0.2,0.2,1])
    GC_init_pos_arr = np.radians(np.array([0,0,0,0,90,0,0]))

    pub_topic = '/dvrk/' + MTM_ARM + '/set_effort_joint'
    sub_topic = '/dvrk/' + MTM_ARM + '/state_joint_current'

    D = 6
    device = 'cpu'

    model = get_model('MTM', use_net, D, device=device)

    model, _, _ = load_model(load_model_path, use_net+'_'+train_type, model)

    model = model.to('cpu')

    pub = rospy.Publisher(pub_topic, JointState, queue_size=15)
    rospy.init_node(MTM_ARM + 'controller', anonymous=True)
    rate = rospy.Rate(10)  # 10hz
    mtm_arm = dvrk.mtm(MTM_ARM)
    count = 0

    # init pose
    mtm_arm.move_joint(GC_init_pos_arr)
    time.sleep(3)
    sub = rospy.Subscriber(sub_topic, JointState, callback)
    while not rospy.is_shutdown():
        pass
    return None

MTM_ARM = 'MTMR'
use_net = 'ReLU_Dual_UDirection'
load_model_path = join("data", "MTMR_28002", "real", "uniform", "N4", 'D6_SinCosInput', "dual","result","model")
train_type = 'PKD'

loop_func(MTM_ARM, use_net, load_model_path, train_type)


