import time
import serial
import numpy as np


def runJTraj(traj, pause_time):
    robot = serial.Serial()
    robot.port = "COM5" # find right port
    robot.baudrate = 9600
    robot.open()

    time.sleep(3)

    print("Moving to start position.")
    # robot.write('{}, {}, {}\n'.format(traj[0,0], traj[0,1], traj[0,2]).encode())
    robot.write(f'{int(traj[0][0])}, {int(traj[0][1])}, {int(traj[0][2])}\n'.encode())
    time.sleep(2)
    print("Running trajectory.")
    print(np.shape(traj), type(traj))
    for i in range(1, len(traj)):
        print(f'{int(traj[i][0])}, {int(traj[i][1])}, {int(traj[i][2])}\n')
        # print(f'{int(traj[i][0])}, {int(traj[i][1])}, {int(traj[i][2])}\n'.encode())
        # robot.write('{}, {}, {}\n'.format(traj[i,0], traj[i,1], traj[i,2]).encode())
        robot.write(f'{traj[i][0]}, {traj[i][1]}, {traj[i][2]}\n'.encode())
        time.sleep(pause_time)

    robot.write("70, 60, 65\n".encode())
    print("Homing.")
    time.sleep(1)

    robot.close()
    print("Port released. Goodbye!")


if __name__ == "__main__":
    from ISTlogo_traj import ISTlogo_traj

    K = 0.216/15 # 0.21/15 s/deg
    a = np.max(np.abs(np.diff(traj, axis=0))) # max traj point distance
    pause_time = K*a*1

    traj = ISTlogo_traj()
    # print(traj[0])
    runJTraj(traj[0], pause_time)
