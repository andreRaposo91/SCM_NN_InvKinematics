from os import startfile
from sksurgerynditracker.nditracker import NDITracker
import ndicapy
import numpy as np
import time
from math import ceil

def parse_strays(stream, str_flag=True):
    num = int(stream[0:2], 16)
    offset = 2 + ceil(num/4)
    # print(num, offset)
    # out_of_vol = int(stream[2:offset], 16)
    i = 0
    pos = []
    # print(stream)
    # print([stream[offset+i*21:offset+i*21+5], stream[offset+i*21+5:offset+i*21+7]])
    if str_flag:
        pos = ['-1234.56, -1234.56, -1234.56'] * num # preallocation
        for i in range(num):
            pos[i] = '[' + stream[offset+i*21:offset+i*21+5] + '.' + stream[offset+i*21+5:offset+i*21+7] + ', ' \
                + stream[offset+i*21+7:offset+i*21+12] + '.' + stream[offset+i*21+12:offset+i*21+14] + ', ' \
                + stream[offset+i*21+14:offset+i*21+19] + '.' + stream[offset+i*21+19:offset+i*21+21] + ']'

    else:
        pos = np.empty((num, 3), dtype=float)
        # print(pos.shape)
        for i in range(num):
            pos[i, 0] = float('.'.join([stream[offset+i*21:offset+i*21+5], stream[offset+i*21+5:offset+i*21+7]]))
            pos[i, 1] = float('.'.join([stream[offset+i*21+7:offset+i*21+12], stream[offset+i*21+12:offset+i*21+14]]))
            pos[i, 2] = float('.'.join([stream[offset+i*21+14:offset+i*21+19], stream[offset+i*21+19:offset+i*21+21]]))
    
    return pos


num = 250

ref_T = np.empty((num, 4, 4), dtype=float)
frames = np.empty((num,), dtype=int)
pos = np.empty((num, 3), float)

# pos = ['-1234.56, -1234.56, -1234.56'] * num

SETTINGS = {
    "tracker type": "polaris",
    "serial_port": "COM6", 
    "romfiles" : ["./polaris/8700449.rom"],
}

tracker = NDITracker(SETTINGS)
device = tracker._device
tracker.start_tracking()
# tracker_connecting.set()
i = 0; misses = 0; threshold = round(num * 0.33)
start_time = time.time()

file = open('test/test_polaris.txt', 'w')

while i < num:

    # _, _, frame_number, tracking, _ = tracker.get_frame()

    # strays_raw = ndicapy.ndiCommand(device, 'TX:1800')

    # if any(np.isnan(tracking[0].flatten())):
    #     # print("Nan in Ref Matrix", i)
    #     misses += 1
    #     # if misses > threshold:
    #     #     break
    #     continue

    # file.write(np.array2string(tracking[0]) + '\n' + parse_strays(strays_raw.split('\n')[1])[0] + '\n')
    # file.flush()

    # ref_T[i] = tracking[0]

    # frames[i] = frame_number[0]

    # pos[i,:] = parse_strays(strays_raw.split('\n')[1], str_flag=False)[0]
    # print(type(parse_strays(strays_raw.split('\n')[1], str_flag=False)[0]))
    # print(parse_strays(strays_raw.split('\n')[1], str_flag=False)[0].shape)
    # time.sleep(0.016667)
    i += 1


time_elapsed = time.time() - start_time


print(f"time_elapsed: {time_elapsed:.4f}s")
if misses > threshold:
    print(f"too many misses - {misses} out of {num} frames")
else:
    print(f"average frame interval: {np.mean(np.diff(frames)):.4f}")

print(frames[:20])
print(np.diff(frames[:20]))

file.close()
