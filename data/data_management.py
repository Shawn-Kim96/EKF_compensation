import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import pickle
import numpy as np
from config.app_config import *
from utils.rotations import Quaternion
import pandas as pd
from datetime import datetime
from data_utils import StampedData
import iso8601
import os
import geomag


class DataPreProcess:
    """
    EKF_Compenstation/data/test_data2.csv 파일을 읽어서 acc, gyro, magnetometer 값을 분리시켜 dictionary 형태로 저장
    # TODO: (철수님) test_data2.csv 가 어떻게 만들어진 데이터인지 설명 부탁드립니다!
    """
    def __init__(self):
        current_dir = os.path.abspath(os.getcwd())
        test_data_path = os.path.join(current_dir, 'test_data2.csv')
        _df = pd.read_csv(test_data_path, encoding='utf-8-sig')
        imu_f = StampedData()
        imu_w = StampedData()
        imu_m = StampedData()

        imu_f.data = np.array(
            _df[['accelerometerAccelerationX', 'accelerometerAccelerationY', 'accelerometerAccelerationZ']])
        imu_f.t = np.array(_df.apply(lambda x: datetime.fromtimestamp(x['loggingTime']), axis=1))
        imu_w.data = np.array(_df[['gyroRotationX', 'gyroRotationY', 'gyroRotationZ']])
        imu_w.t = np.array(_df.apply(lambda x: datetime.fromtimestamp(x['loggingTime']), axis=1))
        imu_m.data = np.array(_df[['magnetometerX', 'magnetometerY', 'magnetometerZ']])
        imu_m.t = np.array(_df.apply(lambda x: datetime.fromtimestamp(x['loggingTime']), axis=1))

        self.data = {'imu_f': imu_f, 'imu_w': imu_w, 'imu_m': imu_m}


class DataManagement:
    """
    IMU 데이터값을 input으로 받으면
        1. 센서 covariance 상수 값을 이용해 각 acc, gps, mgn 오차 값을 넣어준다.
        2. 쿼터니언 값을 계산한다.
    """
    def __init__(self, is_var_time_scale_: bool = False):
        """
        Parameters
        ----------
        is_var_time_scale_ : gps, acc, mgn 등 센서값의 timescale이 다른지 물어보는 변수 / 현재는 테스트를 위해 sample data 사용한다.
        ----------
        설명
        Each element of the data dictionary is stored as an item from the data dictionary, which we
        will store in local variables, described by the following:
            gt: Data object containing ground truth. with the following fields:
                a: Acceleration of the vehicle, in the inertial frame
                v: Velocity of the vehicle, in the inertial frame
                p: Position of the vehicle, in the inertial frame
                alpha: Rotational acceleration of the vehicle, in the inertial frame
                w: Rotational velocity of the vehicle, in the inertial frame
                r: Rotational position of the vehicle, in Euler (XYZ) angles in the inertial frame
                _t: Timestamp in ms.
           imu_f: StampedData object with the imu specific force data (given in vehicle frame).
                data: The actual data
                t: Timestamps in ms.
           imu_w: StampedData object with the imu rotational velocity (given in the vehicle frame).
                data: The actual data
                t: Timestamps in ms.
           imu_m: Stamped Data object with the imu magnetometer
           gnss: StampedData object with the GNSS data.
                data: The actual data
                t: Timestamps in ms.
        """
        if not is_var_time_scale_:
            data = DataPreProcess().data

        # 1. get imu data from original data

        # if is_gnss_: self.gt = data['gt']
        self.imu_f = data['imu_f']  # shape : [time_stamp, 3 (x, y, z)]
        self.imu_w = data['imu_w']
        self.imu_m = data['imu_m']  # shape : [time_stamp, 3 (x, y, z)]
        # if is_gnss_: self.gnss = data['gnss']

        # 2. set sensor uncertainty
        ################################################################################################
        # Now that our data is set up, we can start getting things ready for our solver. One of the
        # most important aspects of a filter is setting the estimated sensor variances correctly.
        # We set the values here.
        ################################################################################################
        self.var_imu_f = VAR_IMU_F
        self.var_imu_w = VAR_IMU_W
        self.var_imu_m = VAR_IMU_M
        self.vars = np.hstack((self.var_imu_f, self.var_imu_m))
        # self.var_gnss = VAR_GNSS

        # 3. set constant values
        ################################################################################################
        # We can also set up some constants that won't change for any iteration of our solver.
        ################################################################################################
        self.g = GRAVITY
        gm = geomag.GeoMag()
        mg = gm.GeoMag(MAG_LAT, MAG_LNG)
        self.r = np.array([np.cos(mg.dec), 0, np.sin(mg.dec)])
        self.gr = np.array([self.g, self.r])

        # self.l_jac = np.zeros([9, 6])
        # self.l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
        # self.h_jac = np.zeros([3, 9])
        # self.h_jac[:, :3] = np.eye(3)  # measurement model jacobian
        #
        # self.l_gyro_jac = np.eye(3)
        # self.h_gyro_jac = np.eye(3)

        # 4. set initial values
        ################################################################################################
        # Let's set up some initial values for our ES-EKF solver.
        ################################################################################################

        # self.p_est = np.zeros([self.imu_f.data.shape[0], 3])  # position estimates
        # self.v_est = np.zeros([self.imu_f.data.shape[0], 3])  # velocity estimates
        self.q_est = np.zeros([self.imu_f.data.shape[0], 4])  # orientation estimates as quaternions
        self.q_mag_est = np.zeros([self.imu_f.data.shape[0], 4])  # orientation estimates as quaternions for magnet
        # self.p_cov = np.zeros([self.imu_f.data.shape[0], 9, 9])  # covariance matrices at each timestep
        self.p_imu_cov = np.zeros([self.imu_f.data.shape[0], 4, 4])  # covariance matrices of gyro at each timestep
        self.p_mag_cov = np.zeros([self.imu_f.data.shape[0], 4, 4])  # covariance matrices of mag at each timestep

        # self.p_est[0] = self.gt.p[0]
        # self.v_est[0] = self.gt.v[0]
        # self.q_est[0] = Quaternion(euler=self.gt.r[0]).to_numpy()
        # self.q_mag_est[0] = Quaternion(euler=self.gt.r[0]).to_numpy()

        # self.p_est[0] = np.array([0, 0, 0])
        # self.v_est[0] = np.array([0, 0, 0])
        # self.q_est[0] = Quaternion(euler=np.array([0, 0, 0])).to_numpy() # [1, 0, 0, 0]
        # self.q_mag_est[0] = Quaternion(euler=np.array([0, 0, 0])).to_numpy()

        # 5. calculate quaternion

        an = self.imu_f.data[0]/np.linalg.norm(self.imu_f.data[0])  # acc normalized
        mn = self.imu_m.data[0]/np.linalg.norm(self.imu_m.data[0])  # magnetic normalized
        c_2 = np.cross(an, mn)
        c_1 = np.cross(c_2, an)

        # c11 = ((self.imu_f.data[0][2]**2)*self.imu_m.data[0][0]) \
        #       - (self.imu_f.data[0][0]*self.imu_f.data[0][2]*self.imu_m.data[0][2]) \
        #       - (self.imu_f.data[0][0]*self.imu_f.data[0][1]*self.imu_m.data[0][1]) \
        #       + ((self.imu_f.data[0][1]**2)*self.imu_m.data[0][0])
        # c21 = ((self.imu_f.data[0][0]**2)*self.imu_m.data[0][1]) \
        #       - (self.imu_f.data[0][0]*self.imu_f.data[0][1]*self.imu_m.data[0][0]) \
        #       - (self.imu_f.data[0][1]*self.imu_f.data[0][2]*self.imu_m.data[0][2]) \
        #       + ((self.imu_f.data[0][2]**2)*self.imu_m.data[0][1])
        # c31 = ((self.imu_f.data[0][1]**2)*self.imu_m.data[0][2]) \
        #       - (self.imu_f.data[0][1]*self.imu_f.data[0][2]*self.imu_m.data[0][1]) \
        #       - (self.imu_f.data[0][0]*self.imu_f.data[0][2]*self.imu_m.data[0][2]) \
        #       + ((self.imu_f.data[0][0]**2)*self.imu_m.data[0][2])
        # c12 = (self.imu_f.data[0][1]*self.imu_m.data[0][2]) - (self.imu_f.data[0][2]*self.imu_m.data[0][1])
        # c22 = (self.imu_f.data[0][2]*self.imu_m.data[0][0]) - (self.imu_f.data[0][0]*self.imu_m.data[0][2])
        # c32 = (self.imu_f.data[0][0]*self.imu_m.data[0][1]) - (self.imu_f.data[0][1]*self.imu_m.data[0][0])

        C = np.array([[c_1[0], c_2[0], an[0]],
                      [c_1[1], c_2[1], an[1]],
                      [c_1[2], c_2[2], an[2]]])
        self.q_est[0] = np.array([[0.5*np.sqrt(C[0][0] + C[1][1] + C[2][2] + 1)],
                                  [0.5*np.sign(C[2][1] - C[1][2])*np.sqrt(C[0][0] - C[1][1] - C[2][2] + 1)],
                                  [0.5*np.sign(C[0][2] - C[2][0])*np.sqrt(C[1][1] - C[2][2] - C[0][0] + 1)],
                                  [0.5*np.sign(C[1][0] - C[0][1])*np.sqrt(C[2][2] - C[0][0] - C[1][1] + 1)]]).reshape(4,)

        # if self.imu_f.data[0][2] >= 0:
        #     self.q_est[0] = np.array([np.sqrt((self.imu_f.data[0][2] + 1)/2),
        #                               -self.imu_f.data[0][1]/np.sqrt(2*(self.imu_f.data[0][2] + 1)),
        #                               self.imu_f.data[0][0]/np.sqrt(2*(self.imu_f.data[0][2] + 1)),
        #                               0])
        # else:
        #     self.q_est[0] = np.array([-self.imu_f.data[0][1]/np.sqrt(2*(1 - self.imu_f.data[0][2])),
        #                               np.sqrt((1 - self.imu_f.data[0][2])/2),
        #                               self.imu_f.data[0][0]/np.sqrt(2*(1 - self.imu_f.data[0][2])),
        #                               0])
        #
        # L = (self.imu_m.data[0][0] ** 2) + (self.imu_m.data[0][1] ** 2)
        # if self.imu_m.data[0][0] >= 0:
        #     self.q_mag_est[0] = np.array([np.sqrt((L + (self.imu_m.data[0][0] * np.sqrt(L)))/2),
        #                                   0,
        #                                   0,
        #                                   self.imu_m.data[0][0]/np.sqrt((2*L) + (self.imu_m.data[0][0] * np.sqrt(L)))])
        # else:
        #     self.q_mag_est[0] = np.array([self.imu_m.data[0][1]/np.sqrt((2*L) - (self.imu_m.data[0][0] * np.sqrt(L))),
        #                                   0,
        #                                   0,
        #                                   np.sqrt((L - (self.imu_m.data[0][0] * np.sqrt(L)))/2)])

        # self.p_cov[0] = np.zeros(9)  # covariance of estimate
        self.p_imu_cov[0] = np.eye(4)  # covariance of estimate in imu

        # self.gnss_i = 0
