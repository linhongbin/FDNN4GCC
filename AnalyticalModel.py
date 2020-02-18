import numpy as np
from math import cos, sin
from torch.utils.data import Dataset
import torch
from sklearn import preprocessing


class MTM_CAD():
    def __init__(self):
        self.g = 9.81
        cm2_x = -0.38
        cm2_y = 0.00
        cm2_z = 0.00
        m2 = 0.65

        cm3_x = -0.25
        cm3_y = 0.00
        cm3_z = 0.00
        m3 = 0.04

        cm4_x = 0.0
        cm4_y = -0.084
        cm4_z = -0.12
        m4 = 0.14

        cm5_x = 0.0
        cm5_y = 0.036
        cm5_z = -0.065
        m5 = 0.04

        cm6_x = 0.0
        cm6_y = -0.025
        cm6_z = 0.05
        m6 = 0.05

        L2 = 0.2794
        L3 = 0.3645
        L4_z0 = 0.1506

        counter_balance = 0.54
        cable_offset = 0.33
        drift2 = -cable_offset
        E5 = 0.007321
        drift5 = - 0.0065

        param_vec = np.zeros((14,1))
        param_vec[1-1, 0] = L2 * m2 + L2 * m3 + L2 * m4 + L2 * m5 + L2 * m6 + cm2_x * m2
        param_vec[2-1, 0] = cm2_y * m2
        param_vec[3-1, 0] = L3 * m3 + L3 * m4 + L3 * m5 + L3 * m6 + cm3_x * m3
        param_vec[4-1, 0] = cm4_y * m4 + cm3_z * m3 + L4_z0 * m4 + L4_z0 * m5 + L4_z0 * m6
        param_vec[5-1, 0] = cm5_z * m5 + cm6_y * m6
        param_vec[6-1, 0] = cm6_z * m6
        param_vec[7-1, 0] = cm4_x * m4
        param_vec[8-1, 0] = - cm4_z * m4 + cm5_y * m5
        param_vec[9-1, 0] = cm5_x * m5
        param_vec[10-1, 0] = cm6_x * m6
        param_vec[11-1, 0] = counter_balance
        param_vec[12-1, 0] = drift2
        param_vec[13-1, 0] = E5
        param_vec[14-1, 0] = drift5
        self.param_vec = param_vec
        self.jnt_upper_limit =  np.radians(np.array([45, 34, 190, 175, 40]))
        self.jnt_lower_limit = np.radians(np.array([-14, -34, -80, -85, -40]))

    def predict(self, input_mat):
        output_mat = np.zeros((input_mat.shape[0], 5))
        for i in range(input_mat.shape[0]):
            q2 = input_mat[i,0]
            q3 = input_mat[i,1]
            q4 = input_mat[i,2]
            q5 = input_mat[i,3]
            q6 = input_mat[i,4]
            R_mat = self.regressor(q2, q3, q4, q5, q6)
            tor = R_mat.dot(self.param_vec).reshape(7)
            output_mat[i,:] = tor[1:-1]
        return output_mat

    def regressor(self, q2, q3, q4, q5, q6):
        g =  self.g
        R_mat = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [g * sin(q2), g * cos(q2), g * cos(q2) * cos(q3) - g * sin(q2) * sin(q3), - g * cos(q2) * sin(q3) - g * cos(q3) * sin(q2), g * cos(q4) * sin(q2) * sin(q3) * sin(q5) - g * cos(q3) * cos(q5) * sin(q2) - g * cos(q2) * cos(q3) * cos(q4) * sin(q5) - g * cos(q2) * cos(q5) * sin(q3), g * cos(q2) * cos(q3) * sin(q4) * sin(q6) + g * cos(q2) * cos(q6) * sin(q3) * sin(q5) + g * cos(q3) * cos(q6) * sin(q2) * sin(q5) - g * sin(q2) * sin(q3) * sin(q4) * sin(q6) + g * cos(q4) * cos(q5) * cos(q6) * sin(q2) * sin(q3) - g * cos(q2) * cos(q3) * cos(q4) * cos(q5) * cos(q6), g * cos(q2) * cos(q3) * cos(q4) - g * cos(q4) * sin(q2) * sin(q3), g * sin(q2) * sin(q3) * sin(q4) - g * cos(q2) * cos(q3) * sin(q4), g * cos(q2) * cos(q3) * cos(q4) * cos(q5) - g * cos(q3) * sin(q2) * sin(q5) - g * cos(q2) * sin(q3) * sin(q5) - g * cos(q4) * cos(q5) * sin(q2) * sin(q3),
                   g * cos(q2) * cos(q3) * cos(q6) * sin(q4) - g * cos(q6) * sin(q2) * sin(q3) * sin(q4) - g * cos(q2) * sin(q3) * sin(q5) * sin(q6) - g * cos(q3) * sin(q2) * sin(q5) * sin(q6) - g * cos(q4) * cos(q5) * sin(q2) * sin(q3) * sin(q6) + g * cos(q2) * cos(q3) * cos(q4) * cos(q5) * sin(q6), 0, 1, 0, 0],
                  [0, 0, g * cos(q2 + q3), -g * sin(q2 + q3), -(g * (2 * cos(q2) * cos(q5) * sin(q3) + 2 * cos(q3) * cos(q5) * sin(q2) + 2 * cos(q2) * cos(q3) * cos(q4) * sin(q5) - 2 * cos(q4) * sin(q2) * sin(q3) * sin(q5))) / 2, (g * (2 * cos(q2) * cos(q3) * sin(q4) * sin(q6) + 2 * cos(q2) * cos(q6) * sin(q3) * sin(q5) + 2 * cos(q3) * cos(q6) * sin(q2) * sin(q5) - 2 * sin(q2) * sin(q3) * sin(q4) * sin(q6) - 2 * cos(q2) * cos(q3) * cos(q4) * cos(q5) * cos(q6) + 2 * cos(q4) * cos(q5) * cos(q6) * sin(q2) * sin(q3))) / 2, (g * (cos(q2 + q3 + q4) + cos(q2 + q3 - q4))) / 2, (g * (2 * sin(q2) * sin(q3) * sin(q4) - 2 * cos(q2) * cos(q3) * sin(q4))) / 2, -(g * (2 * cos(q2) * sin(q3) * sin(q5) + 2 * cos(q3) * sin(q2) * sin(q5) - 2 * cos(q2) * cos(q3) * cos(q4) * cos(q5) + 2 * cos(q4) * cos(q5) * sin(q2) * sin(q3))) / 2,
                   -(g * (2 * cos(q6) * sin(q2) * sin(q3) * sin(q4) - 2 * cos(q2) * cos(q3) * cos(q6) * sin(q4) + 2 * cos(q2) * sin(q3) * sin(q5) * sin(q6) + 2 * cos(q3) * sin(q2) * sin(q5) * sin(q6) - 2 * cos(q2) * cos(q3) * cos(q4) * cos(q5) * sin(q6) + 2 * cos(q4) * cos(q5) * sin(q2) * sin(q3) * sin(q6))) / 2, -cos(q2 + q3), 0, 0, 0],
                  [0, 0, 0, 0, g * sin(q2 + q3) * sin(q4) * sin(q5), g * sin(q2 + q3) * (cos(q4) * sin(q6) + cos(q5) * cos(q6) * sin(q4)), -g * sin(q2 + q3) * sin(q4), -g * sin(q2 + q3) * cos(q4), -g * sin(q2 + q3) * cos(q5) * sin(q4), g * sin(q2 + q3) * (cos(q4) * cos(q6) - cos(q5) * sin(q4) * sin(q6)), 0, 0, 0, 0],
                  [0, 0, 0, 0, -g * (cos(q2) * cos(q3) * sin(q5) - sin(q2) * sin(q3) * sin(q5) + cos(q2) * cos(q4) * cos(q5) * sin(q3) + cos(q3) * cos(q4) * cos(q5) * sin(q2)), g * (cos(q5) * cos(q6) * sin(q2) * sin(q3) - cos(q2) * cos(q3) * cos(q5) * cos(q6) + cos(q2) * cos(q4) * cos(q6) * sin(q3) * sin(q5) + cos(q3) * cos(q4) * cos(q6) * sin(q2) * sin(q5)), 0, 0, -g * (cos(q5) * sin(q2) * sin(q3) - cos(q2) * cos(q3) * cos(q5) + cos(q2) * cos(q4) * sin(q3) * sin(q5) + cos(q3) * cos(q4) * sin(q2) * sin(q5)), -g * (cos(q5) * sin(q2) * sin(q3) * sin(q6) - cos(q2) * cos(q3) * cos(q5) * sin(q6) + cos(q2) * cos(q4) * sin(q3) * sin(q5) * sin(q6) + cos(q3) * cos(q4) * sin(q2) * sin(q5) * sin(q6)), 0, 0, q5, 1],
                  [0, 0, 0, 0, 0, g * (cos(q2) * cos(q6) * sin(q3) * sin(q4) + cos(q3) * cos(q6) * sin(q2) * sin(q4) + cos(q2) * cos(q3) * sin(q5) * sin(q6) - sin(q2) * sin(q3) * sin(q5) * sin(q6) + cos(q2) * cos(q4) * cos(q5) * sin(q3) * sin(q6) + cos(q3) * cos(q4) * cos(q5) * sin(q2) * sin(q6)), 0, 0, 0, g * (cos(q2) * cos(q3) * cos(q6) * sin(q5) - cos(q2) * sin(q3) * sin(q4) * sin(q6) - cos(q3) * sin(q2) * sin(q4) * sin(q6) - cos(q6) * sin(q2) * sin(q3) * sin(q5) + cos(q2) * cos(q4) * cos(q5) * cos(q6) * sin(q3) + cos(q3) * cos(q4) * cos(q5) * cos(q6) * sin(q2)), 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        return R_mat

class MTM_FK():
    def __init__(self):
        self.jnt_upper_limit =  np.radians(np.array([45, 34, 190, 175, 40]))
        self.jnt_lower_limit = np.radians(np.array([-14, -34, -80, -85, -40]))

    def fk_vec(self, q2,q3,q4,q5,q6):
        cm2_x = -0.38
        cm2_y = 0.00
        cm2_z = 0.00
        m2 = 0.65

        cm3_x = -0.25
        cm3_y = 0.00
        cm3_z = 0.00
        m3 = 0.04

        cm4_x = 0.0
        cm4_y = -0.084
        cm4_z = -0.12
        m4 = 0.14

        cm5_x = 0.0
        cm5_y = 0.036
        cm5_z = -0.065
        m5 = 0.04

        cm6_x = 0.0
        cm6_y = -0.025
        cm6_z = 0.05
        m6 = 0.05

        L2 = 0.2794
        L3 = 0.3645
        L4_z0 = 0.1506

        # might not accurate
        L3_parallel = 0
        cm1_x = 0
        cm1_z = 0
        cm1_y = 0

        cm4_parallel_x = 0
        cm4_parallel_y = 0
        cm4_parallel_z = 0

        cm5_parallel_x = 0
        cm5_parallel_y = 0
        cm5_parallel_z = 0

        q1 = 0
        fk_vec = np.array([L2 * sin(q1) * sin(q2) - cm2_z * cos(q1) + cm2_y * cos(q2) * sin(q1) + cm2_x * sin(q1) * sin(q2),
- cm2_z * sin(q1) - L2 * cos(q1) * sin(q2) - cm2_y * cos(q1) * cos(q2) - cm2_x * cos(q1) * sin(q2),
cm2_y * sin(q2) - cm2_x * cos(q2) - L2 * cos(q2),
cm3_y * cos(q1) + L2 * sin(q1) * sin(q2) + L3 * cos(q2) * cos(q3) * sin(q1) + cm3_x * cos(q2) * cos(q3) * sin(q1) - L3 * sin(q1) * sin(q2) * sin(q3) - cm3_z * cos(q2) * sin(q1) * sin(q3) - cm3_z * cos(q3) * sin(q1) * sin(q2) - cm3_x * sin(q1) * sin(q2) * sin(q3),
cm3_y * sin(q1) - L2 * cos(q1) * sin(q2) - L3 * cos(q1) * cos(q2) * cos(q3) - cm3_x * cos(q1) * cos(q2) * cos(q3) + L3 * cos(q1) * sin(q2) * sin(q3) + cm3_z * cos(q1) * cos(q2) * sin(q3) + cm3_z * cos(q1) * cos(q3) * sin(q2) + cm3_x * cos(q1) * sin(q2) * sin(q3),
L3 * sin(q2 + q3) + cm3_z * cos(q2 + q3) + cm3_x * sin(q2 + q3) - L2 * cos(q2),
L2 * sin(q1) * sin(q2) - cm4_z * cos(q1) * cos(q4) + cm4_x * cos(q1) * sin(q4) + L3 * cos(q2) * cos(q3) * sin(q1) - L4_z0 * cos(q2) * sin(q1) * sin(q3) - L4_z0 * cos(q3) * sin(q1) * sin(q2) - L3 * sin(q1) * sin(q2) * sin(q3) - cm4_y * cos(q2) * sin(q1) * sin(q3) - cm4_y * cos(q3) * sin(q1) * sin(q2) + cm4_x * cos(q2) * cos(q3) * cos(q4) * sin(q1) + cm4_z * cos(q2) * cos(q3) * sin(q1) * sin(q4) - cm4_x * cos(q4) * sin(q1) * sin(q2) * sin(q3) - cm4_z * sin(q1) * sin(q2) * sin(q3) * sin(q4),
cm4_x * sin(q1) * sin(q4) - cm4_z * cos(q4) * sin(q1) - L2 * cos(q1) * sin(q2) - L3 * cos(q1) * cos(q2) * cos(q3) + L4_z0 * cos(q1) * cos(q2) * sin(q3) + L4_z0 * cos(q1) * cos(q3) * sin(q2) + L3 * cos(q1) * sin(q2) * sin(q3) + cm4_y * cos(q1) * cos(q2) * sin(q3) + cm4_y * cos(q1) * cos(q3) * sin(q2) - cm4_x * cos(q1) * cos(q2) * cos(q3) * cos(q4) - cm4_z * cos(q1) * cos(q2) * cos(q3) * sin(q4) + cm4_x * cos(q1) * cos(q4) * sin(q2) * sin(q3) + cm4_z * cos(q1) * sin(q2) * sin(q3) * sin(q4),
(cm4_z * cos(q2 + q3 - q4)) / 2 + (cm4_x * sin(q2 + q3 - q4)) / 2 + L4_z0 * cos(q2 + q3) + L3 * sin(q2 + q3) + cm4_y * cos(q2 + q3) - L2 * cos(q2) - (cm4_z * cos(q2 + q3 + q4)) / 2 + (cm4_x * sin(q2 + q3 + q4)) / 2,
cm5_y * cos(q1) * cos(q4) + L2 * sin(q1) * sin(q2) + L3 * cos(q2) * cos(q3) * sin(q1) - L4_z0 * cos(q2) * sin(q1) * sin(q3) - L4_z0 * cos(q3) * sin(q1) * sin(q2) + cm5_x * cos(q1) * cos(q5) * sin(q4) - L3 * sin(q1) * sin(q2) * sin(q3) - cm5_z * cos(q1) * sin(q4) * sin(q5) - cm5_y * cos(q2) * cos(q3) * sin(q1) * sin(q4) - cm5_z * cos(q2) * cos(q5) * sin(q1) * sin(q3) - cm5_z * cos(q3) * cos(q5) * sin(q1) * sin(q2) - cm5_x * cos(q2) * sin(q1) * sin(q3) * sin(q5) - cm5_x * cos(q3) * sin(q1) * sin(q2) * sin(q5) + cm5_y * sin(q1) * sin(q2) * sin(q3) * sin(q4) - cm5_z * cos(q2) * cos(q3) * cos(q4) * sin(q1) * sin(q5) - cm5_x * cos(q4) * cos(q5) * sin(q1) * sin(q2) * sin(q3) + cm5_z * cos(q4) * sin(q1) * sin(q2) * sin(q3) * sin(q5) + cm5_x * cos(q2) * cos(q3) * cos(q4) * cos(q5) * sin(q1),
cm5_y * cos(q4) * sin(q1) - L2 * cos(q1) * sin(q2) - L3 * cos(q1) * cos(q2) * cos(q3) + L4_z0 * cos(q1) * cos(q2) * sin(q3) + L4_z0 * cos(q1) * cos(q3) * sin(q2) + L3 * cos(q1) * sin(q2) * sin(q3) + cm5_x * cos(q5) * sin(q1) * sin(q4) - cm5_z * sin(q1) * sin(q4) * sin(q5) + cm5_y * cos(q1) * cos(q2) * cos(q3) * sin(q4) + cm5_z * cos(q1) * cos(q2) * cos(q5) * sin(q3) + cm5_z * cos(q1) * cos(q3) * cos(q5) * sin(q2) + cm5_x * cos(q1) * cos(q2) * sin(q3) * sin(q5) + cm5_x * cos(q1) * cos(q3) * sin(q2) * sin(q5) - cm5_y * cos(q1) * sin(q2) * sin(q3) * sin(q4) + cm5_x * cos(q1) * cos(q4) * cos(q5) * sin(q2) * sin(q3) - cm5_z * cos(q1) * cos(q4) * sin(q2) * sin(q3) * sin(q5) - cm5_x * cos(q1) * cos(q2) * cos(q3) * cos(q4) * cos(q5) + cm5_z * cos(q1) * cos(q2) * cos(q3) * cos(q4) * sin(q5),
L4_z0 * cos(q2) * cos(q3) - L2 * cos(q2) + L3 * cos(q2) * sin(q3) + L3 * cos(q3) * sin(q2) - L4_z0 * sin(q2) * sin(q3) + cm5_z * cos(q2) * cos(q3) * cos(q5) + cm5_x * cos(q2) * cos(q3) * sin(q5) - cm5_y * cos(q2) * sin(q3) * sin(q4) - cm5_y * cos(q3) * sin(q2) * sin(q4) - cm5_z * cos(q5) * sin(q2) * sin(q3) - cm5_x * sin(q2) * sin(q3) * sin(q5) + cm5_x * cos(q2) * cos(q4) * cos(q5) * sin(q3) + cm5_x * cos(q3) * cos(q4) * cos(q5) * sin(q2) - cm5_z * cos(q2) * cos(q4) * sin(q3) * sin(q5) - cm5_z * cos(q3) * cos(q4) * sin(q2) * sin(q5),
L2 * sin(q1) * sin(q2) + L3 * cos(q2) * cos(q3) * sin(q1) - cm6_x * cos(q1) * cos(q4) * cos(q6) - L4_z0 * cos(q2) * sin(q1) * sin(q3) - L4_z0 * cos(q3) * sin(q1) * sin(q2) - cm6_z * cos(q1) * cos(q4) * sin(q6) - L3 * sin(q1) * sin(q2) * sin(q3) - cm6_y * cos(q1) * sin(q4) * sin(q5) - cm6_z * cos(q1) * cos(q5) * cos(q6) * sin(q4) - cm6_y * cos(q2) * cos(q5) * sin(q1) * sin(q3) - cm6_y * cos(q3) * cos(q5) * sin(q1) * sin(q2) + cm6_x * cos(q1) * cos(q5) * sin(q4) * sin(q6) + cm6_x * cos(q2) * cos(q3) * cos(q6) * sin(q1) * sin(q4) - cm6_y * cos(q2) * cos(q3) * cos(q4) * sin(q1) * sin(q5) + cm6_z * cos(q2) * cos(q3) * sin(q1) * sin(q4) * sin(q6) + cm6_z * cos(q2) * cos(q6) * sin(q1) * sin(q3) * sin(q5) + cm6_z * cos(q3) * cos(q6) * sin(q1) * sin(q2) * sin(q5) - cm6_x * cos(q6) * sin(q1) * sin(q2) * sin(q3) * sin(q4) + cm6_y * cos(q4) * sin(q1) * sin(q2) * sin(q3) * sin(q5) - cm6_x * cos(q2) * sin(q1) * sin(q3) * sin(q5) * sin(q6) - cm6_x * cos(q3) * sin(q1) * sin(q2) * sin(q5) * sin(q6) - cm6_z * sin(q1) * sin(q2) * sin(q3) * sin(q4) * sin(q6) - cm6_z * cos(q2) * cos(q3) * cos(q4) * cos(q5) * cos(q6) * sin(q1) + cm6_x * cos(q2) * cos(q3) * cos(q4) * cos(q5) * sin(q1) * sin(q6) + cm6_z * cos(q4) * cos(q5) * cos(q6) * sin(q1) * sin(q2) * sin(q3) - cm6_x * cos(q4) * cos(q5) * sin(q1) * sin(q2) * sin(q3) * sin(q6),
L4_z0 * cos(q1) * cos(q2) * sin(q3) - L3 * cos(q1) * cos(q2) * cos(q3) - L2 * cos(q1) * sin(q2) + L4_z0 * cos(q1) * cos(q3) * sin(q2) + L3 * cos(q1) * sin(q2) * sin(q3) - cm6_x * cos(q4) * cos(q6) * sin(q1) - cm6_z * cos(q4) * sin(q1) * sin(q6) - cm6_y * sin(q1) * sin(q4) * sin(q5) + cm6_y * cos(q1) * cos(q2) * cos(q5) * sin(q3) + cm6_y * cos(q1) * cos(q3) * cos(q5) * sin(q2) - cm6_z * cos(q5) * cos(q6) * sin(q1) * sin(q4) + cm6_x * cos(q5) * sin(q1) * sin(q4) * sin(q6) - cm6_z * cos(q1) * cos(q2) * cos(q3) * sin(q4) * sin(q6) - cm6_z * cos(q1) * cos(q2) * cos(q6) * sin(q3) * sin(q5) - cm6_z * cos(q1) * cos(q3) * cos(q6) * sin(q2) * sin(q5) + cm6_x * cos(q1) * cos(q6) * sin(q2) * sin(q3) * sin(q4) - cm6_y * cos(q1) * cos(q4) * sin(q2) * sin(q3) * sin(q5) + cm6_x * cos(q1) * cos(q2) * sin(q3) * sin(q5) * sin(q6) + cm6_x * cos(q1) * cos(q3) * sin(q2) * sin(q5) * sin(q6) + cm6_z * cos(q1) * sin(q2) * sin(q3) * sin(q4) * sin(q6) - cm6_x * cos(q1) * cos(q2) * cos(q3) * cos(q6) * sin(q4) + cm6_y * cos(q1) * cos(q2) * cos(q3) * cos(q4) * sin(q5) + cm6_z * cos(q1) * cos(q2) * cos(q3) * cos(q4) * cos(q5) * cos(q6) - cm6_x * cos(q1) * cos(q2) * cos(q3) * cos(q4) * cos(q5) * sin(q6) - cm6_z * cos(q1) * cos(q4) * cos(q5) * cos(q6) * sin(q2) * sin(q3) + cm6_x * cos(q1) * cos(q4) * cos(q5) * sin(q2) * sin(q3) * sin(q6),
L4_z0 * cos(q2) * cos(q3) - L2 * cos(q2) + L3 * cos(q2) * sin(q3) + L3 * cos(q3) * sin(q2) - L4_z0 * sin(q2) * sin(q3) + cm6_y * cos(q2) * cos(q3) * cos(q5) - cm6_y * cos(q5) * sin(q2) * sin(q3) - cm6_z * cos(q2) * cos(q3) * cos(q6) * sin(q5) + cm6_x * cos(q2) * cos(q6) * sin(q3) * sin(q4) + cm6_x * cos(q3) * cos(q6) * sin(q2) * sin(q4) - cm6_y * cos(q2) * cos(q4) * sin(q3) * sin(q5) - cm6_y * cos(q3) * cos(q4) * sin(q2) * sin(q5) + cm6_x * cos(q2) * cos(q3) * sin(q5) * sin(q6) + cm6_z * cos(q2) * sin(q3) * sin(q4) * sin(q6) + cm6_z * cos(q3) * sin(q2) * sin(q4) * sin(q6) + cm6_z * cos(q6) * sin(q2) * sin(q3) * sin(q5) - cm6_x * sin(q2) * sin(q3) * sin(q5) * sin(q6) + cm6_x * cos(q2) * cos(q4) * cos(q5) * sin(q3) * sin(q6) + cm6_x * cos(q3) * cos(q4) * cos(q5) * sin(q2) * sin(q6) - cm6_z * cos(q2) * cos(q4) * cos(q5) * cos(q6) * sin(q3) - cm6_z * cos(q3) * cos(q4) * cos(q5) * cos(q6) * sin(q2),
L3_parallel * cos(q2) * cos(q3) * sin(q1) - cm4_parallel_y * cos(q2) * sin(q1) - cm4_parallel_x * sin(q1) * sin(q2) - cm4_parallel_z * cos(q1) - L3_parallel * sin(q1) * sin(q2) * sin(q3),
cm4_parallel_y * cos(q1) * cos(q2) - cm4_parallel_z * sin(q1) + cm4_parallel_x * cos(q1) * sin(q2) - L3_parallel * cos(q1) * cos(q2) * cos(q3) + L3_parallel * cos(q1) * sin(q2) * sin(q3),
L3_parallel * sin(q2 + q3) + cm4_parallel_x * cos(q2) - cm4_parallel_y * sin(q2),
L3_parallel * cos(q2) * cos(q3) * sin(q1) - cm5_parallel_z * cos(q1) - cm5_parallel_y * cos(q2) * cos(q3) * sin(q1) - L3_parallel * sin(q1) * sin(q2) * sin(q3) - cm5_parallel_x * cos(q2) * sin(q1) * sin(q3) - cm5_parallel_x * cos(q3) * sin(q1) * sin(q2) + cm5_parallel_y * sin(q1) * sin(q2) * sin(q3),
cm5_parallel_y * cos(q1) * cos(q2) * cos(q3) - L3_parallel * cos(q1) * cos(q2) * cos(q3) - cm5_parallel_z * sin(q1) + L3_parallel * cos(q1) * sin(q2) * sin(q3) + cm5_parallel_x * cos(q1) * cos(q2) * sin(q3) + cm5_parallel_x * cos(q1) * cos(q3) * sin(q2) - cm5_parallel_y * cos(q1) * sin(q2) * sin(q3),
L3_parallel * sin(q2 + q3) + cm5_parallel_x * cos(q2 + q3) - cm5_parallel_y * sin(q2 + q3)])
        return fk_vec


    def predict(self, input_mat):
        output_mat = np.zeros((input_mat.shape[0], 21))
        for i in range(input_mat.shape[0]):
            q2 = input_mat[i,0]
            q3 = input_mat[i,1]
            q4 = input_mat[i,2]
            q5 = input_mat[i,3]
            q6 = input_mat[i,4]
            output_mat[i,:] = self.fk_vec(q2,q3,q4,q5,q6)
        return output_mat

    def random_model_sampling(self, sample_num, input_scaler=None, output_scaler=None, is_inputScale = False, is_outputScale = False):
        inputDim = 5
        input_mat = np.zeros((sample_num, inputDim))
        for i in range(sample_num):
            rand_arr = np.random.rand(inputDim)
            input_mat[i,:] = rand_arr*(self.jnt_upper_limit-self.jnt_lower_limit) + self.jnt_lower_limit
        output_mat = self.predict(input_mat)

        if input_scaler is not None:
            input_mat = input_scaler.transform(input_mat)
        elif is_inputScale:
            input_scaler = preprocessing.StandardScaler().fit(input_mat)
            input_mat = input_scaler.transform(input_mat)

        if output_scaler is not None:
            output_mat = output_scaler.transform(output_mat)
        elif is_outputScale:
            output_scaler = preprocessing.StandardScaler().fit(output_mat)
            output_mat = output_scaler.transform(output_mat)

        return input_mat, output_mat, input_scaler, output_scaler




class MTM_MLSE4POL():
    def __init__(self):
        self.g = 9.81
        self.param_vec = np.array([ -0.3049,
        0.1305,
        0.0280,
        0.0127,
        0.0002,
        0.0209,

    -0.0105,
    -0.0018,
    0.0046,
    0.0001,
    0.0335,
    0.0635,
    0.0483,
-0.0070,
-0.0936,
-1.0210,
3.4479,
0,
0,
0,
0.0234,
0.0891,
0.0125,
-0.0311,
0.1426,
0.0492,
0.0415,
-0.0234,
-0.0009,
0.0019,
0.0590,
-0.0968,
-0.0256,
0.0199,
-0.0014,
0.0008,
0.0022,
0.0022,
-0.0017,
0.0177,
0.0120,
0.0683,
0.0068,
0.1208,
0.0699,
-1.0946,
3.4335,
0,
0,
0,
0.0004,
0.0589,
-0.0310,
0.0719,
0.3217,
-0.0508,
0.0500,
-0.0194,
-0.0087,
0.0038,
0.0398,
-0.0951,
-0.0237,
0.0196,
-0.0016,
-0.0010,
-0.0033,
-0.0100,
0.0034,
0.0097]).reshape((70,1))
        self.jnt_upper_limit =  np.radians(np.array([40,45, 34, 190, 175, 40]))
        self.jnt_lower_limit = np.radians(np.array([-40,-14, -34, -80, -85, -40]))
    def regressor_pos(self, q1, q2, q3, q4, q5, q6):
            g = self.g
            R_mat = np.array([[         0,         0,                                     0,                                       0,                                                     0,                                                     0,                                                                                                                             0,                                                                                                                             0,                                                                                                                                                                                                                                     0,                                                                                                                                                                                                                                     0, 1, q1, q1**2, q1**3, q1**4, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[ g*sin(q2), g*cos(q2), g*cos(q2)*cos(q3) - g*sin(q2)*sin(q3), - g*cos(q2)*sin(q3) - g*cos(q3)*sin(q2), g*cos(q2)*cos(q3)*cos(q4) - g*cos(q4)*sin(q2)*sin(q3), g*sin(q2)*sin(q3)*sin(q4) - g*cos(q2)*cos(q3)*sin(q4), g*cos(q4)*sin(q2)*sin(q3)*sin(q5) - g*cos(q3)*cos(q5)*sin(q2) - g*cos(q2)*cos(q3)*cos(q4)*sin(q5) - g*cos(q2)*cos(q5)*sin(q3), g*cos(q2)*cos(q3)*cos(q4)*cos(q5) - g*cos(q3)*sin(q2)*sin(q5) - g*cos(q2)*sin(q3)*sin(q5) - g*cos(q4)*cos(q5)*sin(q2)*sin(q3), g*cos(q2)*cos(q3)*sin(q4)*sin(q6) + g*cos(q2)*cos(q6)*sin(q3)*sin(q5) + g*cos(q3)*cos(q6)*sin(q2)*sin(q5) - g*sin(q2)*sin(q3)*sin(q4)*sin(q6) + g*cos(q4)*cos(q5)*cos(q6)*sin(q2)*sin(q3) - g*cos(q2)*cos(q3)*cos(q4)*cos(q5)*cos(q6), g*cos(q2)*cos(q3)*cos(q6)*sin(q4) - g*cos(q6)*sin(q2)*sin(q3)*sin(q4) - g*cos(q2)*sin(q3)*sin(q5)*sin(q6) - g*cos(q3)*sin(q2)*sin(q5)*sin(q6) - g*cos(q4)*cos(q5)*sin(q2)*sin(q3)*sin(q6) + g*cos(q2)*cos(q3)*cos(q4)*cos(q5)*sin(q6), 0,  0,    0,    0,    0, 1, q2, q2**2, q2**3, q2**4, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[         0,         0, g*cos(q2)*cos(q3) - g*sin(q2)*sin(q3), - g*cos(q2)*sin(q3) - g*cos(q3)*sin(q2), g*cos(q2)*cos(q3)*cos(q4) - g*cos(q4)*sin(q2)*sin(q3), g*sin(q2)*sin(q3)*sin(q4) - g*cos(q2)*cos(q3)*sin(q4), g*cos(q4)*sin(q2)*sin(q3)*sin(q5) - g*cos(q3)*cos(q5)*sin(q2) - g*cos(q2)*cos(q3)*cos(q4)*sin(q5) - g*cos(q2)*cos(q5)*sin(q3), g*cos(q2)*cos(q3)*cos(q4)*cos(q5) - g*cos(q3)*sin(q2)*sin(q5) - g*cos(q2)*sin(q3)*sin(q5) - g*cos(q4)*cos(q5)*sin(q2)*sin(q3), g*cos(q2)*cos(q3)*sin(q4)*sin(q6) + g*cos(q2)*cos(q6)*sin(q3)*sin(q5) + g*cos(q3)*cos(q6)*sin(q2)*sin(q5) - g*sin(q2)*sin(q3)*sin(q4)*sin(q6) + g*cos(q4)*cos(q5)*cos(q6)*sin(q2)*sin(q3) - g*cos(q2)*cos(q3)*cos(q4)*cos(q5)*cos(q6), g*cos(q2)*cos(q3)*cos(q6)*sin(q4) - g*cos(q6)*sin(q2)*sin(q3)*sin(q4) - g*cos(q2)*sin(q3)*sin(q5)*sin(q6) - g*cos(q3)*sin(q2)*sin(q5)*sin(q6) - g*cos(q4)*cos(q5)*sin(q2)*sin(q3)*sin(q6) + g*cos(q2)*cos(q3)*cos(q4)*cos(q5)*sin(q6), 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 1, q3, q3**2, q3**3, q3**4, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[         0,         0,                                     0,                                       0,                               -g*sin(q2 + q3)*sin(q4),                               -g*sin(q2 + q3)*cos(q4),                                                                                                g*sin(q2 + q3)*sin(q4)*sin(q5),                                                                                               -g*sin(q2 + q3)*cos(q5)*sin(q4),                                                                                                                                                                            g*sin(q2 + q3)*(cos(q4)*sin(q6) + cos(q5)*cos(q6)*sin(q4)),                                                                                                                                                                            g*sin(q2 + q3)*(cos(q4)*cos(q6) - cos(q5)*sin(q4)*sin(q6)), 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 1, q4, q4**2, q4**3, q4**4, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[         0,         0,                                     0,                                       0,                                                     0,                                                     0,    -g*(cos(q2)*cos(q3)*sin(q5) - sin(q2)*sin(q3)*sin(q5) + cos(q2)*cos(q4)*cos(q5)*sin(q3) + cos(q3)*cos(q4)*cos(q5)*sin(q2)),    -g*(cos(q5)*sin(q2)*sin(q3) - cos(q2)*cos(q3)*cos(q5) + cos(q2)*cos(q4)*sin(q3)*sin(q5) + cos(q3)*cos(q4)*sin(q2)*sin(q5)),                                                                             g*(cos(q5)*cos(q6)*sin(q2)*sin(q3) - cos(q2)*cos(q3)*cos(q5)*cos(q6) + cos(q2)*cos(q4)*cos(q6)*sin(q3)*sin(q5) + cos(q3)*cos(q4)*cos(q6)*sin(q2)*sin(q5)),                                                                            -g*(cos(q5)*sin(q2)*sin(q3)*sin(q6) - cos(q2)*cos(q3)*cos(q5)*sin(q6) + cos(q2)*cos(q4)*sin(q3)*sin(q5)*sin(q6) + cos(q3)*cos(q4)*sin(q2)*sin(q5)*sin(q6)), 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 1, q5, q5**2, q5**3, q5**4, 0,  0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[         0,         0,                                     0,                                       0,                                                     0,                                                     0,                                                                                                                             0,                                                                                                                             0,         g*(cos(q2)*cos(q6)*sin(q3)*sin(q4) + cos(q3)*cos(q6)*sin(q2)*sin(q4) + cos(q2)*cos(q3)*sin(q5)*sin(q6) - sin(q2)*sin(q3)*sin(q5)*sin(q6) + cos(q2)*cos(q4)*cos(q5)*sin(q3)*sin(q6) + cos(q3)*cos(q4)*cos(q5)*sin(q2)*sin(q6)),         g*(cos(q2)*cos(q3)*cos(q6)*sin(q5) - cos(q2)*sin(q3)*sin(q4)*sin(q6) - cos(q3)*sin(q2)*sin(q4)*sin(q6) - cos(q6)*sin(q2)*sin(q3)*sin(q5) + cos(q2)*cos(q4)*cos(q5)*cos(q6)*sin(q3) + cos(q3)*cos(q4)*cos(q5)*cos(q6)*sin(q2)), 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 1, q6, q6**2, q6**3, q6**4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[         0,         0,                                     0,                                       0,                                                     0,                                                     0,                                                                                                                             0,                                                                                                                             0,                                                                                                                                                                                                                                     0,                                                                                                                                                                                                                                     0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

            return R_mat

    def regressor_neg(self, q1, q2, q3, q4, q5, q6):
        g = self.g
        R_mat = np.array([[         0,         0,                                     0,                                       0,                                                     0,                                                     0,                                                                                                                             0,                                                                                                                             0,                                                                                                                                                                                                                                     0,                                                                                                                                                                                                                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, q1, q1**2, q1**3, q1**4, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0],
[ g*sin(q2), g*cos(q2), g*cos(q2)*cos(q3) - g*sin(q2)*sin(q3), - g*cos(q2)*sin(q3) - g*cos(q3)*sin(q2), g*cos(q2)*cos(q3)*cos(q4) - g*cos(q4)*sin(q2)*sin(q3), g*sin(q2)*sin(q3)*sin(q4) - g*cos(q2)*cos(q3)*sin(q4), g*cos(q4)*sin(q2)*sin(q3)*sin(q5) - g*cos(q3)*cos(q5)*sin(q2) - g*cos(q2)*cos(q3)*cos(q4)*sin(q5) - g*cos(q2)*cos(q5)*sin(q3), g*cos(q2)*cos(q3)*cos(q4)*cos(q5) - g*cos(q3)*sin(q2)*sin(q5) - g*cos(q2)*sin(q3)*sin(q5) - g*cos(q4)*cos(q5)*sin(q2)*sin(q3), g*cos(q2)*cos(q3)*sin(q4)*sin(q6) + g*cos(q2)*cos(q6)*sin(q3)*sin(q5) + g*cos(q3)*cos(q6)*sin(q2)*sin(q5) - g*sin(q2)*sin(q3)*sin(q4)*sin(q6) + g*cos(q4)*cos(q5)*cos(q6)*sin(q2)*sin(q3) - g*cos(q2)*cos(q3)*cos(q4)*cos(q5)*cos(q6), g*cos(q2)*cos(q3)*cos(q6)*sin(q4) - g*cos(q6)*sin(q2)*sin(q3)*sin(q4) - g*cos(q2)*sin(q3)*sin(q5)*sin(q6) - g*cos(q3)*sin(q2)*sin(q5)*sin(q6) - g*cos(q4)*cos(q5)*sin(q2)*sin(q3)*sin(q6) + g*cos(q2)*cos(q3)*cos(q4)*cos(q5)*sin(q6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,    0,    0,    0, 1, q2, q2**2, q2**3, q2**4, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0],
[         0,         0, g*cos(q2)*cos(q3) - g*sin(q2)*sin(q3), - g*cos(q2)*sin(q3) - g*cos(q3)*sin(q2), g*cos(q2)*cos(q3)*cos(q4) - g*cos(q4)*sin(q2)*sin(q3), g*sin(q2)*sin(q3)*sin(q4) - g*cos(q2)*cos(q3)*sin(q4), g*cos(q4)*sin(q2)*sin(q3)*sin(q5) - g*cos(q3)*cos(q5)*sin(q2) - g*cos(q2)*cos(q3)*cos(q4)*sin(q5) - g*cos(q2)*cos(q5)*sin(q3), g*cos(q2)*cos(q3)*cos(q4)*cos(q5) - g*cos(q3)*sin(q2)*sin(q5) - g*cos(q2)*sin(q3)*sin(q5) - g*cos(q4)*cos(q5)*sin(q2)*sin(q3), g*cos(q2)*cos(q3)*sin(q4)*sin(q6) + g*cos(q2)*cos(q6)*sin(q3)*sin(q5) + g*cos(q3)*cos(q6)*sin(q2)*sin(q5) - g*sin(q2)*sin(q3)*sin(q4)*sin(q6) + g*cos(q4)*cos(q5)*cos(q6)*sin(q2)*sin(q3) - g*cos(q2)*cos(q3)*cos(q4)*cos(q5)*cos(q6), g*cos(q2)*cos(q3)*cos(q6)*sin(q4) - g*cos(q6)*sin(q2)*sin(q3)*sin(q4) - g*cos(q2)*sin(q3)*sin(q5)*sin(q6) - g*cos(q3)*sin(q2)*sin(q5)*sin(q6) - g*cos(q4)*cos(q5)*sin(q2)*sin(q3)*sin(q6) + g*cos(q2)*cos(q3)*cos(q4)*cos(q5)*sin(q6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 1, q3, q3**2, q3**3, q3**4, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0],
[         0,         0,                                     0,                                       0,                               -g*sin(q2 + q3)*sin(q4),                               -g*sin(q2 + q3)*cos(q4),                                                                                                g*sin(q2 + q3)*sin(q4)*sin(q5),                                                                                               -g*sin(q2 + q3)*cos(q5)*sin(q4),                                                                                                                                                                            g*sin(q2 + q3)*(cos(q4)*sin(q6) + cos(q5)*cos(q6)*sin(q4)),                                                                                                                                                                            g*sin(q2 + q3)*(cos(q4)*cos(q6) - cos(q5)*sin(q4)*sin(q6)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 1, q4, q4**2, q4**3, q4**4, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0],
[         0,         0,                                     0,                                       0,                                                     0,                                                     0,    -g*(cos(q2)*cos(q3)*sin(q5) - sin(q2)*sin(q3)*sin(q5) + cos(q2)*cos(q4)*cos(q5)*sin(q3) + cos(q3)*cos(q4)*cos(q5)*sin(q2)),    -g*(cos(q5)*sin(q2)*sin(q3) - cos(q2)*cos(q3)*cos(q5) + cos(q2)*cos(q4)*sin(q3)*sin(q5) + cos(q3)*cos(q4)*sin(q2)*sin(q5)),                                                                             g*(cos(q5)*cos(q6)*sin(q2)*sin(q3) - cos(q2)*cos(q3)*cos(q5)*cos(q6) + cos(q2)*cos(q4)*cos(q6)*sin(q3)*sin(q5) + cos(q3)*cos(q4)*cos(q6)*sin(q2)*sin(q5)),                                                                            -g*(cos(q5)*sin(q2)*sin(q3)*sin(q6) - cos(q2)*cos(q3)*cos(q5)*sin(q6) + cos(q2)*cos(q4)*sin(q3)*sin(q5)*sin(q6) + cos(q3)*cos(q4)*sin(q2)*sin(q5)*sin(q6)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 1, q5, q5**2, q5**3, q5**4, 0,  0,    0,    0,    0],
[         0,         0,                                     0,                                       0,                                                     0,                                                     0,                                                                                                                             0,                                                                                                                             0,         g*(cos(q2)*cos(q6)*sin(q3)*sin(q4) + cos(q3)*cos(q6)*sin(q2)*sin(q4) + cos(q2)*cos(q3)*sin(q5)*sin(q6) - sin(q2)*sin(q3)*sin(q5)*sin(q6) + cos(q2)*cos(q4)*cos(q5)*sin(q3)*sin(q6) + cos(q3)*cos(q4)*cos(q5)*sin(q2)*sin(q6)),         g*(cos(q2)*cos(q3)*cos(q6)*sin(q5) - cos(q2)*sin(q3)*sin(q4)*sin(q6) - cos(q3)*sin(q2)*sin(q4)*sin(q6) - cos(q6)*sin(q2)*sin(q3)*sin(q5) + cos(q2)*cos(q4)*cos(q5)*cos(q6)*sin(q3) + cos(q3)*cos(q4)*cos(q5)*cos(q6)*sin(q2)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 1, q6, q6**2, q6**3, q6**4],
[         0,         0,                                     0,                                       0,                                                     0,                                                     0,                                                                                                                             0,                                                                                                                             0,                                                                                                                                                                                                                                     0,                                                                                                                                                                                                                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0, 0,  0,    0,    0,    0]])

        return R_mat
    
    def predict(self, input_mat):
        output_mat = np.zeros((input_mat.shape[0], 5))
        for i in range(input_mat.shape[0]):
            q2 = input_mat[i,0]
            q3 = input_mat[i,1]
            q4 = input_mat[i,2]
            q5 = input_mat[i,3]
            q6 = input_mat[i,4]
            R_mat = (self.regressor_pos(0, q2, q3, q4, q5, q6) + self.regressor_neg(0, q2, q3, q4, q5, q6))/2
            tor = R_mat.dot(self.param_vec).reshape(7)
            output_mat[i,:] = tor[1:-1]
        return output_mat

    def random_model_sampling(self, sample_num, input_scaler=None, output_scaler=None, is_inputScale = False, is_outputScale = False):
        input_mat = np.zeros((sample_num, 5))
        for i in range(sample_num):
            rand_arr = np.random.rand(5)
            input_mat[i,:] = rand_arr*(self.jnt_upper_limit-self.jnt_lower_limit) + self.jnt_lower_limit
        output_mat = self.predict(input_mat)

        if input_scaler is not None:
            input_mat = input_scaler.transform(input_mat)
        elif is_inputScale:
            input_scaler = preprocessing.StandardScaler().fit(input_mat)
            input_mat = input_scaler.transform(input_mat)

        if output_scaler is not None:
            output_mat = output_scaler.transform(output_mat)
        elif is_outputScale:
            output_scaler = preprocessing.StandardScaler().fit(output_mat)
            output_mat = output_scaler.transform(input_mat)

        return input_mat, output_mat, input_scaler, output_scaler

    def predict_SinCosInput(self, q_mat, delta_q_mat):
        D = 6
        output_mat = np.zeros((q_mat.shape[0], D))
        for i in range(q_mat.shape[0]):
            q1 = q_mat[i, 0]
            q2 = q_mat[i,1]
            q3 = q_mat[i,2]
            q4 = q_mat[i,3]
            q5 = q_mat[i,4]
            q6 = q_mat[i,5]
            R_pos = self.regressor_pos(q1, q2, q3, q4, q5, q6)
            R_neg = self.regressor_neg(q1, q2, q3, q4, q5, q6)
            tor_pos = R_pos.dot(self.param_vec).reshape(7)
            tor_neg = R_neg.dot(self.param_vec).reshape(7)
            tor_pos = tor_pos[:-1]
            tor_neg = tor_neg[:-1]
            u = delta_q_mat[i,:]
            tor = tor_pos * u + tor_neg * (1-u)
            output_mat[i,:] = tor
        return output_mat

    def random_sampling_SinCosInput(self, sample_num):
        D = 6
        q_mat = np.zeros((sample_num, D))
        for i in range(sample_num):
            rand_arr = np.random.rand(D)
            q_mat[i, :] = rand_arr * (self.jnt_upper_limit - self.jnt_lower_limit) + self.jnt_lower_limit

        delta_q_mat = np.zeros((sample_num, D))
        for i in range(sample_num):
            rand_arr = np.random.rand(D)
            for j in range(D):
                delta_q_mat[i, j] = 1 if rand_arr[j]>0.5 else 0
        input_mat = np.concatenate((np.sin(q_mat), np.cos(q_mat), delta_q_mat), axis = 1)

        output_mat = self.predict_SinCosInput(q_mat, delta_q_mat)
        return input_mat, output_mat
