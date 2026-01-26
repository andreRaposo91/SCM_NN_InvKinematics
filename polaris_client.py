from math import ceil
import time, datetime, sys, os
import serial
import numpy as np
import matplotlib.pyplot as plt
from sksurgerynditracker.nditracker import NDITracker
import ndicapy
from point_clouds import generate_circle, generate_random_points, generate_square, vert_up_down_seq, vert_to_point_seq, curv_up_down_seq, curv_test, grid, add_neighbours
from data_functions import parse_dataset, polaris2base
from kinematics_functions import invKspace_car, len2jtheta, T_beModule, jtheta2len

def parse_strays(stream, str_flag=True):
    num = int(stream[0:2], 16)
    offset = 2 + ceil(num/4)
    # print(offset)
    # out_of_vol = int(stream[2:offset], 16)
    i = 0
    pos = []
    # print([stream[offset+i*21:offset+i*21+5], stream[offset+i*21+5:offset+i*21+7]])
    if str_flag:
        pos = ['-1234.56, -1234.56, -1234.56'] * num # preallocation
        for i in range(num):
            pos[i] = '[' + stream[offset+i*21:offset+i*21+5] + '.' + stream[offset+i*21+5:offset+i*21+7] + ', ' \
                + stream[offset+i*21+7:offset+i*21+12] + '.' + stream[offset+i*21+12:offset+i*21+14] + ', ' \
                + stream[offset+i*21+14:offset+i*21+19] + '.' + stream[offset+i*21+19:offset+i*21+21] + ']'
            
    return pos
    # else:
    #     pos = [[0., 0., 0.]] * num
    #     for i in range(num):
    #         pos[i] = [
    #             float('.'.join([stream[offset+i*21:offset+i*21+5], stream[offset+i*21+5:offset+i*21+7]])),
    #             float('.'.join([stream[offset+i*21+7:offset+i*21+12], stream[offset+i*21+12:offset+i*21+14]])),
    #             float('.'.join([stream[offset+i*21+14:offset+i*21+19], stream[offset+i*21+19:offset+i*21+21]])),
    #         ]
    #     return pos


if __name__ == "__main__":
    SETTINGS = {
        "tracker type": "polaris",
        "serial_port": "COM3", 
        "romfiles" : ["./polaris/8700449.rom"],
        }
    # sys.exit(0)

    # num_points = 1100
    # traj = generate_random_points(25, 155, num_points, 20)
    # traj = np.insert(traj, 0, [65, 65, 65], axis=0)
    # traj = np.insert(traj, 0, [1220, 1220, 1220], axis=0)
    # traj = vert_up_down_seq(750, 2250, [9,7,10,8])
    # traj = vert_to_point_seq(750, 2250, 8)
    # traj = curv_up_down_seq(750, 2250, 6, 2)
    traj = curv_test(800, 2200, move_num=6, servos=[1,2,0], fixed_pos=[1000, 1220, 1600], reps=1); dataset_type = "_curv2"; pause_time=2.5
    # traj_grid, _, _ = parse_dataset(filename=filename)
    # traj = add_neighbours(traj_grid, 3, 50, 500, grid=False)
    # print(len(traj))

    # num_points = 110
    # traj = generate_random_points(750, 2250, num_points, 150)
    # traj = np.insert(traj, 0, [950, 950, 950], axis=0) # pyright: ignore
    # dataset_type = ""
    
    # num_points = 120
    # traj = add_neighbours(generate_random_points(750, 2250, num_points, 150), 3, 250, 1500)
    # traj = np.insert(traj, 0, [950, 950, 950], axis=0) # pyright: ignore
    # dataset_type = "_" + "rand3N"
    
    # filename = "./data/dataset_10_check_2024-01-17T160944.txt"
    # traj, _, _ = parse_dataset(filename)
    # dataset_type = "_" + "check"; folder = "data"; pause_time=3
    # log_run = True
    # # folder = "check_val"

    folder = "data"

    # filename = "./data/dataset_10_check_2024-01-17T160944.txt"
    # # filename = "./data/dataset_10_check_2024-04-08T143519.txt"
    # og_traj, ref_T, pos = parse_dataset(filename)
    # traj_, pos_base, _ = polaris2base(og_traj, ref_T, pos)
    # pred_traj_ = fnn3_inv_kin(pos_base)
    # traj = (len2jtheta(pred_traj_)).astype('int')
    # print("avg ref err:", sum(np.sum(np.abs(og_traj - traj), axis=0) / len(traj)) / 3)
    # dataset_type = "_" + "inv_fnn3_check"

    # filename = "./data/dataset_10_check_2024-01-17T160944.txt"
    # # filename = "./data/dataset_10_check_2024-04-08T143519.txt"
    # og_traj, ref_T, pos = parse_dataset(filename)
    # traj_, pos_base, _ = polaris2base(og_traj, ref_T, pos)
    # traj = np.array([invKspace_car(*p, theta_flag=True) for p in pos_base])
    # print("avg ref err:", sum(np.sum(np.abs(og_traj - traj), axis=0) / len(traj)) / 3)
    # dataset_type = "_" + "inv_pcc_check"
    
    # points_gen_str = "generate_square(16, 80, (0, 0, 100), (0, 0, 0))"; pause_time = .15; first_point_split = 12; folder="cont_val_square"
    # points_gen_str = "generate_square(8, 80, (0, 0, 100), (0, 0, 0))"; pause_time = .18; first_point_split = 6; folder="cont_val_square"
    # points_gen_str = "generate_square(8, 80, (0, 0, 100), (0, 0, 0))"; pause_time = 2.25; first_point_split = 2; folder="val"; test ="square"
    # points_gen_str = "generate_square(8, 45, (0, 0, 110), (0, 45, 0))"; pause_time = 2.25; first_point_split = 2; folder="val"; test ="square"
    # points_gen_str = "generate_square(8, 45, (0, 0, 110), (0, 45, 0))"; pause_time = 0.15; first_point_split = 4; folder="cont_val_square"
    
    # points_gen_str = "generate_circle(60, 20, (30, 35, 105), (10, 80, 45))"; pause_time = 0.2; first_point_split=6; folder="cont_val_circle2"; test="circle"
    # points_gen_str = "generate_circle(40, 20, (30, 35, 105), (10, 80, 45))"; pause_time = 0.75; first_point_split=3; folder="val_circle2"; test="circle"
    # points_gen_str = "generate_circle(30, 50, (0, 0, 100), (0, 0, 0))"; pause_time = 0.15; first_point_split=8; folder="cont_val_circle"; test="circle"
    # points_gen_str = "generate_circle(40, 35, (30, 20, 100), (0, 45, 45))"; pause_time = 0.3; first_point_split=7; folder="cont_val_circle"; test="circle"
    # points_gen_str = "generate_circle(40, 50, (0, 0, 100), (0, 0, 0))"; pause_time = 0.75; first_point_split=3; folder="val_circle"; test="circle"
    # points_gen_str = "generate_circle(20, 35, (30, 20, 100), (0, 45, 45))"; pause_time = 1.5; first_point_split=6; folder="val_circle"; test="circle"

    # traj_points = eval(points_gen_str)
    
    # log_run = False
    log_run = True
    
    plot = False
    plot = True

    # folder = "val";
    # from inv_kin_val import fnn3_inv_kin
    # pred_traj_ = fnn3_inv_kin(traj_points)
    # traj = (len2jtheta(pred_traj_)).astype('int')
    # traj = np.insert(traj, 0, [traj[0] - (traj[0] - [1220, 1220, 1220]) / first_point_split * (first_point_split - i) for i in range(1, first_point_split)], axis=0)
    # # print("avg ref err:", sum(np.sum(np.abs(og_traj - traj), axis=0) / len(traj)) / 3)
    # dataset_type = "_" + "inv_fnn3_" + test

    # first_point_split = 2
    # folder = "val"
    # # log_run = True
    # from inv_kin_val import fnn6_inv_kin
    # pred_traj_ = fnn6_inv_kin(traj_points)
    # traj = (len2jtheta(pred_traj_)).astype('int')
    # traj = np.insert(traj, 0, [traj[0] - (traj[0] - [1220, 1220, 1220]) / first_point_split * (first_point_split - i) for i in range(1, first_point_split)], axis=0)
    # # print("avg ref err:", sum(np.sum(np.abs(og_traj - traj), axis=0) / len(traj)) / 3)
    # dataset_type = "_" + "inv_fnn6_" + test
    
    # folder = "val"
    # log_run = True
    # from inv_kin_val import rnn_inv_kin
    # traj_points = np.insert(traj_points, 0, T_beModule(jtheta2len(np.ones((3,))*1220), [], 0, 0)[:3,3], axis=0)
    # pred_traj_ = rnn_inv_kin(traj_points)
    # traj = (len2jtheta(pred_traj_)).astype('int')
    # traj = np.insert(traj, 0, [traj[0] - (traj[0] - [1220, 1220, 1220]) / first_point_split * (first_point_split - i) for i in range(1, first_point_split)], axis=0)
    # # print("avg ref err:", sum(np.sum(np.abs(og_traj - traj), axis=0) / len(traj)) / 3)
    # dataset_type = "_" + "inv_rnn_" + test

    # first_point_split = 2
    # folder = "val"
    # log_run = True
    # traj = np.array([invKspace_car(*p, theta_flag=True) for p in traj_points] * 2)
    # # # traj = np.insert(traj, 0, [1220, 1220, 1220], axis=0)
    # traj = np.insert(traj, 0, [traj[0] - (traj[0] - [1220, 1220, 1220]) / first_point_split * (first_point_split - i) for i in range(first_point_split)], axis=0)
    # # print("avg ref err:", sum(np.sum(np.abs(og_traj - traj), axis=0) / len(traj)) / 3)
    # dataset_type = "_" + "inv_pcc_" + test

    # print(traj)
    
    if plot:
        plt.plot(traj)
        plt.hlines([750, 2250], [0]*2, [len(traj)]*2)
        print(traj.shape)
        print(traj)
        plt.show()
        sys.exit(0)
    
    tracker = NDITracker(SETTINGS)
    device = tracker._device

    dt = datetime.datetime.now(datetime.timezone.utc).isoformat().split('.')[0].replace(':', '')
    # print(datetime.datetime.now(datetime.timezone.utc))

    # file = open(f'./data/dataset_{len(traj)}{dataset_type}_{dt}.txt', 'w')
    # print(f'./data/dataset_{len(traj)}{dataset_type}_{dt}.txt')
    file = open(f'./{folder}/dataset_{len(traj)}{dataset_type}_{dt}.txt', 'w')
    print(f'./{folder}/dataset_{len(traj)}{dataset_type}_{dt}.txt')

    tracker.start_tracking()
    capture_mask = [True] * len(traj)
    # capture_mask = [False, True] * len(traj)

    # tracker._capture_string = 'TX:0801'
    robot = serial.Serial()
    robot.port = "COM4" # find right port
    robot.baudrate = 9600
    robot.open()

    time.sleep(3)
    # time.sleep(pause_time)

    # robot.write('{}, {}, {}\n'.format(traj[0,0], traj[0,1], traj[0,2]).encode())
    # print("Moving to start position.")
    print("Running trajectory.")
    # print(np.shape(traj), type(traj))
    # sys.exit(0)
    while robot.in_waiting > 1:
        robot.readline()
        continue
    # for i in range(1, len(traj)):
    start_time = time.time()
    for i in range(len(traj)):
        robot.write(f'{traj[i][0]}, {traj[i][1]}, {traj[i][2]}\n'.encode())
        # print(f'{traj[i][0]}, {traj[i][1]}, {traj[i][2]}\n'.encode())
        time.sleep(pause_time)
        # port_handles, timestamps, framenumbers, tracking, quality = tracker.get_frame()
        if capture_mask[i]:
            error = 1
            try:
                _, _, _, tracking, _ = tracker.get_frame()
                error = 2
                strays_raw = ndicapy.ndiCommand(device, 'TX:1800')
                # print(tracking)
                # print(np.array2string(tracking[0]), '\n', parse_strays(strays_raw.split('\n')[1]))
                file.write(f'{int(traj[i][0])}, {int(traj[i][1])}, {int(traj[i][2])}\n')
                if any(np.isnan(tracking[0].flatten())): print("Nan in Ref Matrix", i)
                file.write(np.array2string(tracking[0]) + '\n' + parse_strays(strays_raw.split('\n')[1])[0] + '\n')
                file.flush()
            except Exception as e:
                print('error at point', i, f'\nError at command {error}:', e)
                continue

    robot.write("1220, 1220, 1220\n".encode())
    print("Homing.")
    time.sleep(1)

    robot.close()
    print("Port released. Goodbye!")

    tracker.stop_tracking()

    tracker.close()
    file.close()

    # if log_run:
        # elapsed_time = time.time() - start_time
        # with open(f"./{folder}/run_log.txt", 'a') as log_file:
            # log_file.write(f"timestamp: {dt}; traj command: {points_gen_str}; inv_kin_model: {dataset_type[1:]};pause_time: {pause_time}\n") # ;first_point_split: {first_point_split}
            # log_file.write(f"timestamp: {dt}; traj command: {points_gen_str}; inv_kin_model: {dataset_type[1:]};pause_time: {pause_time};elapsed_time: {elapsed_time:.3f}\n") # ;first_point_split: {first_point_split}

    if not log_run:
        os.remove(f'./{folder}/dataset_{len(traj)}{dataset_type}_{dt}.txt')

