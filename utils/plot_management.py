def _plot_result(self):
    est_traj_fig = plt.figure()
    ax = est_traj_fig.add_subplot(111, projection='3d')
    ax.plot(self.p_est[:, 0], self.p_est[:, 1], self.p_est[:, 2], label='Estimated')
    if IS_VAR_TIME_SCALE: ax.plot(self.gt.p[:, 0], self.gt.p[:, 1], self.gt.p[:, 2], label='Ground Truth')
    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Northing [m]')
    ax.set_zlabel('Up [m]')
    ax.set_title('Ground Truth and Estimated Trajectory')
    if IS_VAR_TIME_SCALE:
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 200)
        ax.set_zlim(-2, 2)
        ax.set_xticks([0, 50, 100, 150, 200])
        ax.set_yticks([0, 50, 100, 150, 200])
        ax.set_zticks([-2, -1, 0, 1, 2])

    ax.legend(loc=(0.62, 0.77))
    ax.view_init(elev=45, azim=-50)
    plt.show()


def _plot_error(self):
    error_fig, ax = plt.subplots(2, 3)
    error_fig.suptitle('Error Plots')
    titles = ['Easting', 'Northing', 'Up', 'Roll', 'Pitch', 'Yaw']
    for i in range(3):
        ax[0, i].plot(range(self.num_m), self.gt.p[:, i] - self.p_est[:self.num_m, i])
        ax[0, i].plot(range(self.num_m), 3 * self.p_cov_std[:self.num_m, i], 'r--')
        ax[0, i].plot(range(self.num_m), -3 * self.p_cov_std[:self.num_m, i], 'r--')
        ax[0, i].set_title(titles[i])
    ax[0, 0].set_ylabel('Meters')

    for i in range(3):
        ax[1, i].plot(range(self.num_m), \
                      angle_normalize(self.gt.r[:, i] - self.p_est_euler[:self.num_m, i]))
        ax[1, i].plot(range(self.num_m), 3 * self.p_cov_euler_std[:self.num_m, i], 'r--')
        ax[1, i].plot(range(self.num_m), -3 * self.p_cov_euler_std[:self.num_m, i], 'r--')
        ax[1, i].set_title(titles[i + 3])
    ax[1, 0].set_ylabel('Radians')
    plt.show()