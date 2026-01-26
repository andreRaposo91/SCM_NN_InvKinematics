import numpy as np
import matplotlib.pyplot as plt

from kinematics_functions import invKspace

def ISTlogo_traj():
    xoff = -19
    yoff = 66
    scale = 2.1
    x = xoff + scale * np.array([2, 4.25, 6.5, 8.75, 11, 13.25, 15.5, 17.75, 20, 20, 20, 20, 20, 19.5, 18.5, 17, 15.2, 13.2, 11, 8.8, 6.8, 5, 3.5, 2.5, 2, 2, 2, 2, 2])
    y = yoff + scale * np.array([22, 22, 22, 22, 22, 22, 22, 22, 22, 19.25, 16.5, 13.75, 11, 8.7, 6.3, 4.3, 2.5, 1.1, 0, 1.1, 2.5, 4.3, 6.3, 8.7, 11, 13.75, 16.5, 19.25, 22])
    xI = xoff + scale * np.array([7.4, 7.4, 7.4, 7.4, 7.4])
    yI = yoff + scale * np.array([16.1, 14.3, 12.5, 10.75, 9])
    xT = xoff + scale * np.array([13.6, 14.6, 15.6, 14.6, 14.6, 14.6, 14.6, 14.6])
    yT = yoff + scale * np.array([16.1, 16.1, 16.1, 16.1, 14.3, 12.5, 10.75, 9])
    xS = xoff + scale * np.array([7.4, 8.1, 9.2, 10.3, 11, 11, 11, 11, 11, 11, 11, 11.7, 12.8, 13.9, 14.6])
    yS = yoff + scale * np.array([6.7, 5.8, 5.5, 5.8, 6.7, 8.6, 10.5, 12.5, 14.5, 16.5, 18.5, 19.4, 19.7, 19.4, 18.5])

    # plt.figure(2)
    # plt.plot(x, y, '*-')
    # plt.axis('equal')
    # plt.plot(xI, yI, '*-')
    # plt.plot(xT, yT, '*-')
    # plt.plot(xS, yS, '*-')

    traj1 = invKspace(x[::-1], y[::-1], np.pi * 0)
    traj2 = invKspace(xI, yI, np.pi * 0)
    traj3 = invKspace(xS, yS, np.pi * 0)
    traj4 = invKspace(xT, yT, np.pi * 0)

    s = 3
    traj = np.vstack([np.array([65, 65, 65]), np.tile([65, 70, 60], (s, 1))])

    # Define the data sets (traj1, traj2, traj3, traj4) as a list
    data_sets = [traj1, traj2, traj3, traj4]

    for data_set in data_sets:
        # Append initial data
        traj = np.vstack([traj, np.tile(data_set[0] + [0, 5, -5], (s, 1))])
        # Append the data set
        traj = np.vstack([traj, data_set])
        # Append final data
        traj = np.vstack([traj, np.tile(data_set[-1] + [0, 5, -5], (s, 1))])

    # traj = np.round(traj[:, [1, 2, 0]])

    # print(np.min(traj))
    # print(np.max(traj))

    def jtheta2len(p): # = lambda p:
        return (p - 316.1) / -2.468

    traj_ = jtheta2len(traj)

    # plt.figure(9)
    # plt.plot(traj, '-o', label=['1', '2', '3'])
    # plt.legend()

    plt.show()

    return (traj, traj_)


if __name__ == "__main__":
    traj, traj_ = ISTlogo_traj()
    print('len:', len(traj))
    print(traj_)
    # print(traj[0][:10])
    # print(traj[1][:10])
