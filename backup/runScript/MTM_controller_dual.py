import rospy
from sensor_msgs.msg import JointState
from evaluateTool import predictList
import dvrk
import numpy as np
from os.path import join
import torch
from Net import *
from loadModel import get_model, load_model
import time


MTM_ARM = 'MTMR'
pub_topic = '/dvrk/' + MTM_ARM + '/set_effort_joint'
sub_topic = '/dvrk/' + MTM_ARM + '/state_joint_current'
train_data_path = join("data", "MTMR_28002",'real', 'uniform', 'D5N5','')
use_net = 'Dual_Vanilla_SinSigmoidNet'
D = 5
device = 'cpu'

modelList = get_model('MTM', use_net, D, device=device)

modelList, input_scalerList, output_scalerList = load_model('.','test_Controller_list',modelList)

for model in modelList:
    model = model.to('cpu')

pub = rospy.Publisher(pub_topic, JointState, queue_size=15)
rospy.init_node(MTM_ARM + 'controller', anonymous=True)
rate = rospy.Rate(10)  # 10hz
mtm_arm = dvrk.mtm(MTM_ARM)
count = 0


def callback(data):
    global use_net
    global D
    global input_scalerList
    global output_scalerList
    global pub
    global count

    start = time.clock()


    pos_list = []
    effort_list = []
    for i in range(6):
        pos_list.append(data.position[i])
        effort_list.append(data.effort[i])
    pos_arr = np.array(pos_list)
    effort_arr = np.array(effort_list)
    if D == 5:
        # only get joint input from Joint 2 to 6
        pos_arr = pos_arr[1:]
    #print(pos_arr)
    input = torch.from_numpy(pos_arr).to('cpu').float()
    input = input.unsqueeze(0)
    #print(input)
    output = predictList(modelList, input, input_scalerList, output_scalerList)
    output = output.squeeze(0)
    output_arr = output.numpy()
    if (count == 50):
        print('predict:', output_arr)
        print('measure:',effort_arr[1:6])
        print('error:',output_arr-effort_arr[1:6])
        count = 0
    else:
        count = count+1
    #print(count)

    msg = JointState()
    msg.effort = []
    msg.effort.append(0)
    msg.effort.append(output_arr[0])
    msg.effort.append(output_arr[1])
    msg.effort.append(output_arr[2])
    msg.effort.append(output_arr[3])
    msg.effort.append(output_arr[4])
    msg.effort.append(0)

    pub.publish(msg)
    elapsed = time.clock()
    elapsed = elapsed - start
    #print "Time spent in (function name) is: ", elapsed




if __name__ == '__main__':
    # init pose
    init_pos = np.array([0.0, 0.0, 0.0, 0.0, 3.1415/2.0, 0.0, 0.0])
    mtm_arm.move_joint(init_pos)
    time.sleep(3)
    sub = rospy.Subscriber(sub_topic, JointState, callback)
    while not rospy.is_shutdown():
        pass
