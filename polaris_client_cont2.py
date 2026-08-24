from math import ceil
import time, datetime, sys, os
from pandas._libs.tslibs import timestamps
import serial
import numpy as np
# from sksurgerynditracker.nditracker import NDITracker
# import ndicapy
from point_clouds import (
    generate_coil,
    generate_circle,
    generate_random_points,
    generate_square,
    vert_up_down_seq,
    vert_to_point_seq,
    curv_up_down_seq,
    curv_test,
    grid,
    add_neighbours,
)
from data_functions import parse_dataset, polaris2base, parse_cont_dataset
from val_plots_functions import parse_log
from kinematics_functions import invKspace_car, len2jtheta, T_beModule, jtheta2len
from draw_functions import draw_robot
import asyncio

# import matplotlib
# # print(matplotlib.get_backend())
# matplotlib.rcParams.update(matplotlib.rcParamsDefault)
# matplotlib.use("TKAgg", force=True)
import matplotlib.pyplot as plt

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

def find_traj(run_folder, points_gen_str, model):
    return None
    if isinstance(run_folder, list):
        for fold in run_folder:
            traj = find_traj(fold, points_gen_str, model)
            if traj is not None:
                return traj

    elif isinstance(run_folder, str):
        if os.path.exists(run_folder) and os.path.exists(os.path.join(run_folder, "run_log.txt")):
            filename = os.path.join(run_folder, "run_log.txt")
            timestamp = ""
            with open(filename, 'r') as run_file:
                # run_lines = run_file.readlines()
                for line in run_file.readlines():
                    if model in line and points_gen_str in line:
                        # print(line.split(' ')[1][:-1])
                        timestamp = line.split(' ')[1][:-1]
                        break

            if not timestamp:
                print("points_gen_str or model not found")
                return None

            datafile = ""
            for df in os.listdir(run_folder):
                if timestamp in df:
                    datafile = df

            if not datafile:
                print("datafile with timestamp", timestamp, "not found")
                return None

            datafile = os.path.join(run_folder, datafile)
            print(datafile)
                    
            if "cont" in datafile:
                return parse_cont_dataset(datafile)[0]

            else:
                return parse_dataset(datafile)

        else:
            print("no 'run_log.txt' in 'run_folder'")
            return None

    else:
        print("find_traj: invalid 'run_folder' format")
        return None

async def polaris_track(file, file_lock, tracker_connecting, serial_port="COM6"):
    global in_traj, count

    SETTINGS = {
        "tracker type": "polaris",
        "serial_port": serial_port, 
        "romfiles" : ["./polaris/8700449.rom"],
    }
    
    tracker = NDITracker(SETTINGS)
    device = tracker._device
    tracker.start_tracking()
    tracker_connecting.set()

    while in_traj:
        try:
            _, _, _, tracking, _ = tracker.get_frame()

            strays_raw = ndicapy.ndiCommand(device, 'TX:1800')
        except Exception as e:
            print("failed at collecting from polaris:\n", e)
            continue

        # if any(np.isnan(tracking[0].flatten())):
        #     print("Nan in Ref Matrix", i)
        #     continue
        
        try:
            async with file_lock:
                file.write(np.array2string(tracking[0]) + '\n' + parse_strays(strays_raw.split('\n')[1])[0] + '\n')
                file.flush()
                count += 1
        except Exception as e:
            print("failed writing to file:\n", e)
            continue

        await asyncio.sleep(0)
        # await asyncio.sleep(0.017)
    
    tracker.stop_tracking()
    tracker.close()

async def main():
    global in_traj, count

    # filename = "./data/dataset_10_check_2024-01-17T160944.txt"
    # traj, _, _ = parse_dataset(filename)
    # dataset_type = "_" + "check"; folder = "data"; pause_time=3
    # log_run = False
    
    # points_gen_str = "generate_square(16, 80, (0, 0, 100), (0, 0, 0))"; pause_time = .15; first_point_split = 12; folder="cont_val_square"
    # points_gen_str = "generate_square(8, 90, (0, 0, 100), (0, 0, 0), start_point=(0,0,125.5))"; pause_time = 0.18; folder="cont_val/cont_square"; test="square"
    # points_gen_str = "generate_square(8, 80, (0, 0, 100), (0, 0, 0))"; pause_time = 2.25; first_point_split = 2; folder="val"; test ="square"
    # points_gen_str = "generate_square(8, 45, (0, 0, 110), (0, 45, 0))"; pause_time = 2.25; folder="val"; test ="square"
    # points_gen_str = "generate_square(8, 70, (10, 0, 105), (1, 25, 45), start_point=(0, 0, 125.5, 3))"; pause_time = 0.75; folder="cont_val/square2"; test ="square"
    # # arduino stap delay: 2000
    points_gen_str = "generate_square(8, 70, (10, 0, 105), (1, 25, 45), start_point=(0, 0, 125.5, 4))"; pause_time = 0.23; folder="cont_val/cont_square2"; test ="square"
    
    # points_gen_str = "generate_circle(30, 50, (0, 0, 100), (0, 0, 0), start_point=(0, 0, 125.5, 3))"; pause_time = 0.75; folder="cont_val/circle"; test="circle"
    # # arduino step delay: 2500
    points_gen_str = "generate_circle(30, 50, (0, 0, 100), (0, 0, 0), start_point=(0, 0, 125.5, 6))"; pause_time = 0.15; folder="cont_val/cont_circle"; test="circle"
    
    # points_gen_str = "generate_circle(25, 35, (0, 20, 110), (0, 45, 90), start_point=(0, 0, 125.5, 3))"; pause_time = 0.75; folder="cont_val/circle2"; test="circle"
    # # arduino step delay: 1700
    # points_gen_str = "generate_circle(25, 35, (0, 20, 110), (0, 45, 90), start_point=(0, 0, 125.5, 5))"; pause_time = 0.27; folder="cont_val/cont_circle2"; test="circle"
    
    # points_gen_str = "generate_circle(35, 20, (30, 35, 105), (10, 80, 45))"; pause_time = 0.2; folder="cont_val/cont_circle3"; test="circle" 
    # points_gen_str = "generate_circle(35, 20, (30, 35, 105), (10, 80, 45))"; pause_time = 0.75; folder="cont_val/circle3"; test="circle"

    # points_gen_str = "generate_coil(60, 14, 100, 3, starting_point=(-60, 0, 105), rotations=(0, 90, 0), starts=(0, 0, 125.5, 3), spread_xy=1.5)"; pause_time = 0.75; folder="cont_val/coil"; test="coil"
    # # arduino step delay: 2000
    points_gen_str = "generate_coil(60, 14, 100, 3, starting_point=(-60, 0, 105), rotations=(0, 90, 0), starts=(0, 0, 125.5, 6), spread_xy=1.5)"; pause_time = 0.16; folder="cont_val/cont_coil"; test="coil"

    traj_points = eval(points_gen_str)
    
    log_run = False
    # log_run = True
    
    plot = False
    plot = True

    model = 'pcc'
    # model = 'fnn3'
    # model = 'fnn6'
    # model = 'rnn'
    # model = 'fnn3_pcc'
    # model = 'fnn6_pcc'
    # model = 'rnn_pcc'

    # possible_folders = ["./cont_val/circle", "./cont_val/cont_circle"]
    # possible_folders = ["./cont_val/circle2", "./cont_val/cont_circle2"]
    possible_folders = folder
    # possible_folders = ["./cont_val/circle3", "./cont_val/cont_circle3"]

    # traj = np.insert(traj, 0, [traj[0] - (traj[0] - [1220, 1220, 1220]) / first_point_split * (first_point_split - i) for i in range(1, first_point_split)], axis=0)
    
    if model == 'fnn3':
        if (traj := find_traj(possible_folders, points_gen_str, model)) is None:
            from inv_kin_val import fnn3_inv_kin
            pred_traj_ = fnn3_inv_kin(traj_points)
            traj = (len2jtheta(pred_traj_)).astype('int')
            # print("avg ref err:", sum(np.sum(np.abs(og_traj - traj), axis=0) / len(traj)) / 3)
        else:
            print("Found 'traj'\n", traj[:5])
        dataset_type = "_" + "inv_fnn3_" + test

    elif model == 'fnn6':
        if (traj := find_traj(possible_folders, points_gen_str, model)) is None:
            from inv_kin_val import fnn6_inv_kin
            pred_traj_ = fnn6_inv_kin(traj_points)
            traj = (len2jtheta(pred_traj_)).astype('int')
            # print("avg ref err:", sum(np.sum(np.abs(og_traj - traj), axis=0) / len(traj)) / 3)
        else:
            print("Found 'traj'")
        dataset_type = "_" + "inv_fnn6_" + test
    
    elif model == 'rnn':
        if (traj := find_traj(possible_folders, points_gen_str, model)) is None:
            from inv_kin_val import rnn_inv_kin
            traj_points = np.insert(traj_points, 0, T_beModule(jtheta2len(np.ones((3,))*1220), [], 0, 0)[:3,3], axis=0)
            pred_traj_ = rnn_inv_kin(traj_points)
            traj = (len2jtheta(pred_traj_)).astype('int')
        else:
            print("Found 'traj'")
            # print("avg ref err:", sum(np.sum(np.abs(og_traj - traj), axis=0) / len(traj)) / 3)
        dataset_type = "_" + "inv_rnn_" + test

    elif model == 'fnn3_pcc':
        if (traj := find_traj(possible_folders, points_gen_str, model)) is None:
            from inv_kin_val import fnn3_pcc_inv_kin
            pred_traj_ = fnn3_pcc_inv_kin(traj_points)
            traj = (len2jtheta(pred_traj_)).astype('int')
            # print("avg ref err:", sum(np.sum(np.abs(og_traj - traj), axis=0) / len(traj)) / 3)
        else:
            print("Found 'traj'\n", traj[:5])
        dataset_type = "_" + "inv_fnn3-pcc_" + test
    
    elif model == 'fnn6_pcc':
        if (traj := find_traj(possible_folders, points_gen_str, model)) is None:
            from inv_kin_val import fnn6_pcc_inv_kin
            pred_traj_ = fnn6_pcc_inv_kin(traj_points)
            traj = (len2jtheta(pred_traj_)).astype('int')
            # print("avg ref err:", sum(np.sum(np.abs(og_traj - traj), axis=0) / len(traj)) / 3)
        else:
            print("Found 'traj'\n", traj[:5])
        dataset_type = "_" + "inv_fnn6-pcc_" + test
    
    elif model == 'rnn_pcc':
        if (traj := find_traj(possible_folders, points_gen_str, model)) is None:
            from inv_kin_val import rnn_pcc_inv_kin
            traj_points = np.insert(traj_points, 0, T_beModule(jtheta2len(np.ones((3,))*1220), [], 0, 0)[:3,3], axis=0)
            pred_traj_ = rnn_pcc_inv_kin(traj_points)
            traj = (len2jtheta(pred_traj_)).astype('int')
            # print("avg ref err:", sum(np.sum(np.abs(og_traj - traj), axis=0) / len(traj)) / 3)
        else:
            print("Found 'traj'\n", traj[:5])
        dataset_type = "_" + "inv_rnn-pcc_" + test

    else:
        pred_traj_ = np.array([invKspace_car(*p, theta_flag=False) for p in traj_points])
        traj = (len2jtheta(pred_traj_)).astype('int')
        # print(traj[:5])
        # # traj = np.insert(traj, 0, [1220, 1220, 1220], axis=0)
        # print("avg ref err:", sum(np.sum(np.abs(og_traj - traj), axis=0) / len(traj)) / 3)
        dataset_type = "_" + "inv_pcc_" + test

    if plot:
        plt.figure(figsize=(12,5)).tight_layout(pad=3)
        plt.gcf().suptitle(f"Planned Trajectory - {test.capitalize()}")
        plt.gcf().add_subplot(121, projection="3d")
        draw_robot(plt.gca(), alpha_mult=0.65)
        plt.gca().plot(*traj_points.T, marker='.')
        plt.gca().scatter(*traj_points[0].T, label="Start", color="green")
        plt.gca().scatter(*traj_points[-1].T, label="Finish", color="red")
        plt.gca().legend()
        if test == 'coil':
            plt.xlim([-50, 50])
        plt.gca().set_zlim((0, 145))
        plt.gca().set_aspect('equal')

        # plt.figure()
        plt.gcf().add_subplot(122)
        # plt.gca().plot(traj, label=['1', '2', '3'])
        # plt.gca().hlines([750, 2250], [0]*2, [len(traj)]*2, color='k', linestyle='--', label="limits")
        # plt.gca().legend(title="Servo References")
        plt.gca().plot(pred_traj_, label=['1', '2', '3'])
        plt.gca().hlines([86, 144], [0]*2, [len(traj)]*2, color='k', linestyle='--', label="limits")
        plt.gca().legend(title="Flexible Rods", loc="upper right", bbox_to_anchor=(1.28, 1))
        plt.gca().set_title("Flexible Rod Lengths along trajectory")
        plt.ylabel("Flexible Rod Length [mm]")
        plt.xlabel("Trajectory Points")
        # print(len(traj))
        # print(traj[:10], traj[-4:])
        # traj = [invKspace_car(*p, theta_flag=True) for p in traj_points]
        # print(len(traj))
        # print(traj[:4], traj[-4:])
        plt.show()
        exit = input("'x' for exit, nothing to continue: ") # sys.exit(0)
        if exit == 'x':
            sys.exit(0)
    sys.exit()

    if any([any([True for val in np.array(traj)[:,i] if val > 2250 or val < 750]) for i in range(3)]):
        sys.exit("Error: Reference out of Bounds")

    dt = datetime.datetime.now(datetime.timezone.utc).isoformat().split('.')[0].replace(':', '')

    if not os.path.exists(folder):
        os.mkdir(folder)
    file = open(f'./{folder}/cont_dataset_{len(traj)}{dataset_type}_{dt}.txt', 'w')
    print(f'./{folder}/cont_dataset_{len(traj)}{dataset_type}_{dt}.txt')

    # capture_mask = [True] * len(traj)

    robot = serial.Serial()
    robot.port = "COM5"
    robot.baudrate = 9600
    robot.open()

    time.sleep(1.5)

    while robot.in_waiting > 1:
        robot.readline()
        continue

    in_traj = True
    file_lock = asyncio.Lock()
    tracker_connecting = asyncio.Event()
    count = 0

    track_task = asyncio.create_task(polaris_track(file, file_lock, tracker_connecting))

    await asyncio.sleep(0)  # Ensure the event loop starts processing tasks

    start_time = time.time()
    print("Running trajectory.")
    for i in range(len(traj)):
        robot.write(f'{traj[i][0]}, {traj[i][1]}, {traj[i][2]}\n'.encode())
        # if capture_mask[i]:
        await asyncio.sleep(pause_time)
        try:
            async with file_lock:
                file.write(f'--{count}--\n{int(traj[i][0])}, {int(traj[i][1])}, {int(traj[i][2])}\n')
                file.flush()
                count = 0
        except Exception as e:
            print('error at point', i, '\nError:', e)
            continue

    await asyncio.sleep(pause_time)
    in_traj = False
    elapsed_time = time.time() - start_time
    robot.write("1220, 1220, 1220\n".encode())
    print("Homing.")
    time.sleep(1)

    robot.close()
    print("Port released. Goodbye!")

    file.close()

    if log_run:
        with open(f"./{folder}/run_log.txt", 'a') as log_file:
            log_file.write(f"timestamp: {dt}; traj command: {points_gen_str}; inv_kin_model: {dataset_type[1:]};pause_time: {pause_time};elapsed_time: {elapsed_time:.3f}\n")
            print(f"Logged in './{folder}/run_log.txt'")

    if not log_run:
        os.remove(f'./{folder}/cont_dataset_{len(traj)}{dataset_type}_{dt}.txt')
        print("Deleting runfile")

if __name__ == "__main__":
    asyncio.run(main())
