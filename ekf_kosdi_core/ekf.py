import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import matmul
from data.data_management import *
from utils.rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion
# from ekf_core.complementary_filter import estimate_orientation


class EKF(DataManagement):
    def __init__(self):
        super().__init__(is_var_time_scale_=IS_VAR_TIME_SCALE)
        if IS_VAR_TIME_SCALE:
            self.num_m = self.imu_m.data.shape[0]
        self.p_est_euler = []
        self.p_cov_euler_std = []
        self.p_cov_std = None

    def measurement_update(self, field, measurement, var, q_check, p_cov_check):
        ######################################################
        # field: acc or mag field
        # measurement: elements measured from IMU
        # var: variance of IMU
        # q_check: q_est[k]
        # p_cov_check: p_imu_cov[k]
        ######################################################

        # 3.1 Compute Kalman Gain
        # K_k = P_k * H_k.T * inv( H_k * P_k * H_k.T + R_k )

        ######################################################
        # Def of H_gyro: [del(a_dot)/del(q)]
        # ref1: https://ahrs.readthedocs.io/en/latest/filters/ekf.html#prediction-step
        # ref2: https://m.blog.naver.com/enewltlr/220918689039
        ######################################################

        h = 2 * np.array([])
        h = 2 * np.array([[(field[0] * (0.5 - (q_check[2] ** 2) + (q_check[3] ** 2))) + (field[1] * ((q_check[0] * q_check[3]) + (q_check[1] * q_check[2]))) + (field[2] * ((q_check[1] * q_check[3]) - (q_check[0] * q_check[2])))],
                          [(field[0] * ((q_check[1] * q_check[2]) - (q_check[0] * q_check[3]))) + (field[1] * (0.5 - (q_check[1] ** 2) - (q_check[3] ** 2))) + (field[2] * ((q_check[0] * q_check[1]) + (q_check[2] * q_check[3])))],
                          [(field[0] * ((q_check[0] * q_check[2]) + (q_check[1] * q_check[3]))) + (field[1] * ((q_check[2] * q_check[3]) - (q_check[0] * q_check[1]))) + (field[2] * (0.5 - (q_check[1] ** 2) - (q_check[2] ** 2)))]])
        H = 2 * np.array([[(field[1] * q_check[3]) - (field[2] * q_check[2]), (field[1] * q_check[2]) + (field[2] * q_check[3]), -(2 * field[0] * q_check[2]) + (field[1] * q_check[1]) - (field[2] * q_check[0]), -(2 * field[0] * q_check[2]) + (field[1] * q_check[0]) + (field[2] * q_check[1])],
                          [-(field[0] * q_check[3]) + (field[2] * q_check[1]), (field[0] * q_check[2]) - (2 * field[1] * q_check[1]) + (field[2] * q_check[0]), (field[0] * q_check[1]) + (field[2] * q_check[3]), -(field[2] * q_check[0]) - (2 * field[1] * q_check[3]) + (field[2] * q_check[2])],
                          [(field[0] * q_check[2]) - (field[1] * q_check[1]), (field[0] * q_check[3]) - (field[1] * q_check[0]) - (2 * field[2] * q_check[1]), (field[0] * q_check[0]) + (field[1] * q_check[3]) - (2 * field[2] * q_check[2]), (field[0] * q_check[1]) + (field[1] * q_check[2])]])
        R = np.eye(3)
        np.fill_diagonal(R, var)

        try:
            '''
            Description: np.linalg.inv(matmul(h_jac, matmul(p_cov_check, h_jac.T)) + sensor_var ))) 
            '''
            S_inv = np.linalg.inv(matmul(H, matmul(p_cov_check, H.T)) + R)
            # print("temp: ", temp.shape, "sensor_var: ", sensor_var)
            K = matmul(p_cov_check, matmul(H.T, S_inv))

        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                raise "A singular matrix "

        # 3.2 Compute error state
        # print("y_k size: ", y_k.shape, "h_jac size: ", h_jac.shape, "p_check size: ", p_check.shape, "P_CHECK: ", p_check)
        '''
        Originally, p_check must be replaced to H_k * P_k + sensor_noise
        '''
        # 3.3 Correct predicted state
        v = measurement.reshape((3, 1)) - h
        q_hat = q_check + matmul(K, v).reshape(4,)
        # q_gyro_hat = Quaternion(axis_angle=matmul(K_gyro, v)).quat_mult_right(q_gyro_check)

        # 3.4 Compute corrected covariance
        p_cov_hat = matmul(np.eye(4) - matmul(K, H), p_cov_check)

        return q_hat, p_cov_hat

    def measurement_update_mag_added(self, field, measurement, var, q_check, p_cov_check):
        h = 2 * np.array([[(field[0][0] * (0.5 - (q_check[2] ** 2) + (q_check[3] ** 2))) + (field[0][1] * ((q_check[0] * q_check[3]) + (q_check[1] * q_check[2]))) + (field[0][2] * ((q_check[1] * q_check[3]) - (q_check[0] * q_check[2])))],
                          [(field[0][0] * ((q_check[1] * q_check[2]) - (q_check[0] * q_check[3]))) + (field[0][1] * (0.5 - (q_check[1] ** 2) - (q_check[3] ** 2))) + (field[0][2] * ((q_check[0] * q_check[1]) + (q_check[2] * q_check[3])))],
                          [(field[0][0] * ((q_check[0] * q_check[2]) + (q_check[1] * q_check[3]))) + (field[0][1] * ((q_check[2] * q_check[3]) - (q_check[0] * q_check[1]))) + (field[0][2] * (0.5 - (q_check[1] ** 2) - (q_check[2] ** 2)))],
                          [(field[1][0] * (0.5 - (q_check[2] ** 2) + (q_check[3] ** 2))) + (field[1][1] * ((q_check[0] * q_check[3]) + (q_check[1] * q_check[2]))) + (field[1][2] * ((q_check[1] * q_check[3]) - (q_check[0] * q_check[2])))],
                          [(field[1][0] * ((q_check[1] * q_check[2]) - (q_check[0] * q_check[3]))) + (field[1][1] * (0.5 - (q_check[1] ** 2) - (q_check[3] ** 2))) + (field[1][2] * ((q_check[0] * q_check[1]) + (q_check[2] * q_check[3])))],
                          [(field[1][0] * ((q_check[0] * q_check[2]) + (q_check[1] * q_check[3]))) + (field[1][1] * ((q_check[2] * q_check[3]) - (q_check[0] * q_check[1]))) + (field[1][2] * (0.5 - (q_check[1] ** 2) - (q_check[2] ** 2)))]
                          ])
        H = 2 * np.array([[(field[0][1] * q_check[3]) - (field[0][2] * q_check[2]), (field[0][1] * q_check[2]) + (field[0][2] * q_check[3]), -(2 * field[0][0] * q_check[2]) + (field[0][1] * q_check[1]) - (field[0][2] * q_check[0]), -(2 * field[0][0] * q_check[2]) + (field[0][1] * q_check[0]) + (field[0][2] * q_check[1])],
                          [-(field[0][0] * q_check[3]) + (field[0][2] * q_check[1]), (field[0][0] * q_check[2]) - (2 * field[0][1] * q_check[1]) + (field[0][2] * q_check[0]), (field[0][0] * q_check[1]) + (field[0][2] * q_check[3]), -(field[0][2] * q_check[0]) - (2 * field[0][1] * q_check[3]) + (field[0][2] * q_check[2])],
                          [(field[0][0] * q_check[2]) - (field[0][1] * q_check[1]), (field[0][0] * q_check[3]) - (field[0][1] * q_check[0]) - (2 * field[0][2] * q_check[1]), (field[0][0] * q_check[0]) + (field[0][1] * q_check[3]) - (2 * field[0][2] * q_check[2]), (field[0][0] * q_check[1]) + (field[0][1] * q_check[2])],
                          [(field[1][1] * q_check[3]) - (field[1][2] * q_check[2]), (field[1][1] * q_check[2]) + (field[1][2] * q_check[3]), -(2 * field[1][0] * q_check[2]) + (field[1][1] * q_check[1]) - (field[1][2] * q_check[0]), -(2 * field[1][0] * q_check[2]) + (field[1][1] * q_check[0]) + (field[1][2] * q_check[1])],
                          [-(field[1][0] * q_check[3]) + (field[1][2] * q_check[1]), (field[1][0] * q_check[2]) - (2 * field[1][1] * q_check[1]) + (field[1][2] * q_check[0]), (field[1][0] * q_check[1]) + (field[1][2] * q_check[3]), -(field[1][2] * q_check[0]) - (2 * field[1][1] * q_check[3]) + (field[1][2] * q_check[2])],
                          [(field[1][0] * q_check[2]) - (field[1][1] * q_check[1]), (field[1][0] * q_check[3]) - (field[1][1] * q_check[0]) - (2 * field[1][2] * q_check[1]), (field[1][0] * q_check[0]) + (field[1][1] * q_check[3]) - (2 * field[1][2] * q_check[2]), (field[1][0] * q_check[1]) + (field[1][1] * q_check[2])]
                          ])
        R = np.eye(6)
        np.fill_diagonal(R, var)

        try:
            S_inv = np.linalg.inv(matmul(H, matmul(p_cov_check, H.T)) + R)
            K = matmul(p_cov_check, matmul(H.T, S_inv))
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                raise "A singular matrix "

        v = measurement.reshape((6, 1)) - h
        q_hat = q_check + matmul(K, v).reshape(4,)

        p_cov_hat = matmul(np.eye(4) - matmul(K, H), p_cov_check)

        return q_hat, p_cov_hat

    def _conv_est_quat2euler(self):
        for i in range(len(self.q_est)):
            qc = Quaternion(*self.q_est[i, :])
            self.p_est_euler.append(qc.to_euler())

            # First-order approximation of RPY covariance
            J = rpy_jacobian_axis_angle(qc.to_axis_angle())
            self.p_cov_euler_std.append(np.sqrt(np.diagonal(J @ self.p_cov[i, 6:, 6:] @ J.T)))

        self.p_est_euler = np.array(self.p_est_euler)
        self.p_cov_euler_std = np.array(self.p_cov_euler_std)

        # Get uncertainty estimates from P matrix
        self.p_cov_std = np.sqrt(np.diagonal(self.p_imu_cov[:, :3, :3], axis1=1, axis2=2))

    def ekf_main(self):
        for k in range(1, self.imu_f.data.shape[0]):
            if IS_VAR_TIME_SCALE:
                delta_t = self.imu_f.t[k] - self.imu_f.t[k - 1]
            else:
                delta_t = (self.imu_f.t[k] - self.imu_f.t[k - 1]).astype('timedelta64[ns]').astype(float)/100000000

            # 1. Update state with IMU inputs
            ################ CORRECTION STEP ###########################################################################
            Omega_k = np.array([[0, -self.imu_w.data[k - 1][0], -self.imu_w.data[k - 1][1], -self.imu_w.data[k - 1][2]],
                                [self.imu_w.data[k - 1][0], 0, self.imu_w.data[k - 1][2], -self.imu_w.data[k - 1][1]],
                                [self.imu_w.data[k - 1][1], -self.imu_w.data[k - 1][2], 0, self.imu_w.data[k - 1][0]],
                                [self.imu_w.data[k - 1][2], self.imu_w.data[k - 1][1], -self.imu_w.data[k - 1][0], 0]])

            ############################################################################################################
            # Equation: q_est_k = f(q_est_k-1, w_t) = (I + 0.5*delta_t*W_t)*q_k-1
            ############################################################################################################
            # self.q_est[k] = Quaternion(euler=delta_t * self.imu_w.data[k - 1]).quat_mult_right(self.q_est[k - 1])
            self.q_est[k] = matmul((np.eye(4) + (0.5 * delta_t * Omega_k)), self.q_est[k - 1])

            # 1.1 Linearize the motion model and compute Jacobians
            ############################################################################################################
            # Def of F: F(q_k-1, w_k) \
            # ref1: https://ahrs.readthedocs.io/en/latest/filters/ekf.html#prediction-step
            # ref2: https://m.blog.naver.com/enewltlr/220918689039
            ############################################################################################################
            F = np.array([[1, -0.5 * delta_t * self.imu_w.data[k - 1][0], 0.5 * delta_t * self.imu_w.data[k - 1][1], 0.5 * delta_t * self.imu_w.data[k - 1][2]],
                          [0.5 * delta_t * self.imu_w.data[k - 1][0], 1, 0.5 * delta_t * self.imu_w.data[k - 1][2], -0.5 * delta_t * self.imu_w.data[k - 1][1]],
                          [0.5 * delta_t * self.imu_w.data[k - 1][1], -0.5 * delta_t * self.imu_w.data[k - 1][2], 1, 0.5 * delta_t * self.imu_w.data[k - 1][0]],
                          [0.5 * delta_t * self.imu_w.data[k - 1][2], 0.5 * delta_t * self.imu_w.data[k - 1][1], -0.5 * delta_t * self.imu_w.data[k - 1][0], 1]])

            # 2. Propagate uncertainty
            ############################################################################################################
            # Def of W: del(Omega(q_k-1, w_k))/del(w) \
            # ref1: https://ahrs.readthedocs.io/en/latest/filters/ekf.html#prediction-step
            # ref2: https://m.blog.naver.com/enewltlr/220918689039
            ############################################################################################################
            W = 0.5 * delta_t * np.array([[-self.q_est[k - 1][1], -self.q_est[k - 1][2], -self.q_est[k - 1][3]],
                                          [self.q_est[k - 1][0], -self.q_est[k - 1][3], self.q_est[k - 1][2]],
                                          [self.q_est[k - 1][3], self.q_est[k - 1][0], -self.q_est[k - 1][1]],
                                          [-self.q_est[k - 1][2], self.q_est[k - 1][1], self.q_est[k - 1][0]]])
            std_w = self.var_imu_w * np.eye(3)

            self.p_imu_cov[k] = matmul(F, matmul(self.p_imu_cov[k - 1], F.T)) + matmul(W, matmul(std_w, W.T))

            # 3. Check availability of GNSS measurements
            ################ PREDICTION STEP ###########################################################################
            vars = np.hstack((self.var_imu_f, self.var_imu_m))
            if IS_VAR_TIME_SCALE:
                for i in range(len(self.imu_m.t)):
                    if abs(self.imu_m.t[i] - self.imu_f.t[k]) < 0.01:
                        am = np.hstack((self.imu_f.data[k], self.imu_m.data[i]))
                        q_est_from_gyro, p_cov_from_gyro = \
                            self.measurement_update(self.g, self.imu_f.data[k], self.var_imu_f, self.q_est[k],
                                                    self.p_imu_cov[k])
                        q_est_from_mag, p_cov_from_mag = \
                            self.measurement_update_mag_added(self.gr, am, vars, self.q_est[k], self.p_imu_cov[k])
                        # q_est_from_mag, p_cov_from_mag = \
                        #     self.measurement_update(self.r, self.imu_m.data[i], self.var_imu_m, q_est_from_gyro,
                        #                             p_cov_from_gyro)

            else:
                am = np.hstack((self.imu_f.data[k], self.imu_m.data[k]))
                q_est_from_gyro, p_cov_from_gyro =\
                    self.measurement_update(self.g, self.imu_f.data[k], self.var_imu_f, self.q_est[k], self.p_imu_cov[k])
                q_est_from_mag, p_cov_from_mag = \
                    self.measurement_update_mag_added(self.gr, am, vars, self.q_est[k], self.p_imu_cov[k])
                # q_est_from_mag, p_cov_from_mag = \
                #     self.measurement_update(self.r, self.imu_m.data[k], self.var_imu_m, q_est_from_gyro, p_cov_from_gyro)

                delta_q_est_from_gyro = np.array([0, 0, Quaternion(*q_est_from_gyro).to_euler()[2]]) \
                                        - np.array([0, 0, Quaternion(*self.q_est[k - 1]).to_euler()[2]])
                delta_q_est_from_mag = np.array([0, 0, Quaternion(*q_est_from_mag).to_euler()[2]]) \
                                        - np.array([0, 0, Quaternion(*self.q_est[k - 1]).to_euler()[2]])
                diff_q_gyro_btw_mag = delta_q_est_from_mag - delta_q_est_from_gyro

                if abs(diff_q_gyro_btw_mag[2]) < 2*THRESHOLD:
                    self.q_est[k] = q_est_from_mag
                    self.p_imu_cov[k] = p_cov_from_mag
                else:
                    self.q_est[k] = q_est_from_gyro
                    self.p_imu_cov[k] = p_cov_from_gyro

        # Results and Analysis #########################################################################

        ################################################################################################
        '''
        Now that we have state estimates for all of our sensor data, let's plot the results. This plot
        will show the ground truth and the estimated trajectories on the same plot. Notice that the
        estimated trajectory continues past the ground truth. This is because we will be evaluating
        your estimated poses from the part of the trajectory where you don't have ground truth!
        '''
        ################################################################################################

        ################################################################################################
        '''
        We can also plot the error for each of the 6 DOF, with estimates for our uncertainty
        included. The error estimates are in blue, and the uncertainty bounds are red and dashed.
        The uncertainty bounds are +/- 3 standard deviations based on our uncertainty (covariance).
        '''
        ################################################################################################
        self._conv_est_quat2euler()

