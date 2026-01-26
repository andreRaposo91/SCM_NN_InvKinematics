import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from data_functions import parse_dataset, polaris2base

traj, ref_T, pos = parse_dataset("./data/dataset_1049_rand3N_2024-02-07T144446.txt")
traj_, pos_base, est_pos_base = polaris2base(traj, ref_T, pos)

points_list = pos_base[1:100]
pcc_list = est_pos_base[1:len(points_list)+1]

fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='3d')

# ax1 = fig1.add_subplot(1, 2, 1, projection='3d')
# ax2 = fig1.add_subplot(1, 2, 2, projection='3d')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_aspect('equal')
ax1.set_xlim(-100, 100)
ax1.set_ylim(-100, 100)
ax1.set_zlim(0, 150)
ax1.set_title('Trajectory with Neighboring Points')

# Initialize scatter and plot objects
rp_scatters = []
n_scatters = []
rp_cons = []
n_cons = []
rp_pcc_cmp = [None] * 5
n_cmp = [None] * 3

def update1(frame):
    global rp_scatters, n_scatters, rp_cons, n_cons, rp_pcc_cmp

    # Clear oldest plot elements
    if len(rp_scatters) > 1:
        rp_scatters[0].remove()
        n_scatters[0].remove()
        rp_cons[0][0].remove()
        # print(n_cons[0][0].set(visible=False))
        # print(n_cons[0][0].remove())
        n_cons[0][0][0].remove()
        n_cons[0][1][0].remove()
        n_cons[0][2][0].remove()

        del rp_scatters[0]
        del n_scatters[0]
        del rp_cons[0]
        del n_cons[0]
        
        # print(len(n_cons[0]))

    # Plot current frame
    i = frame * 7
    rp = points_list[i]
    rp_pcc = pcc_list[i]
    rp_scatter = ax1.scatter(rp[0], rp[1], rp[2], c='red', label='Random Points')
    neighbors = points_list[i + 1:i + 7:2]
    n_scatter = ax1.scatter(neighbors[:, 0], neighbors[:, 1], neighbors[:, 2], c='blue', label='Neighboring Points')
    # for neighbor in neighbors:
    #     n_con = ax1.plot([rp[0], neighbor[0]], [rp[1], neighbor[1]], [rp[2], neighbor[2]], c='blue')
    
    n_cons.append([ax1.plot([rp[0], neighbor[0]], [rp[1], neighbor[1]], [rp[2], neighbor[2]], c='blue', linestyle='dotted') for neighbor in neighbors])
    if i + 7 < len(points_list):
        rp_con = ax1.plot([rp[0], points_list[i + 7, 0]], [rp[1], points_list[i + 7, 1]], [rp[2], points_list[i + 7, 2]], c='red', linestyle='dashed')
        rp_cons.append(rp_con)

    # Save current plot elements
    rp_scatters.append(rp_scatter)
    n_scatters.append(n_scatter)

    # Set ax1is labels

    
    # if i > 0:
    #     # rp_pcc_cmp[0]._offsets3d = (rp_pcc[0], rp_pcc[1], rp_pcc[2])
    #     ax2.clear()
    #     rp_pcc_cmp[0].remove()
    #     # rp_pcc_cmp[0] = None
    #     rp_pcc_cmp[0] = ax2.scatter(rp_pcc[0], rp_pcc[1], rp_pcc[2], label='pcc_est', marker='*')
    #     for j in range(4):
    #         rp_pcc_cmp[1+j].remove()
    #         # rp_pcc_cmp[1+j] = None
    #         # rp_pcc_cmp[1+j]._offsets3d = (points_list[i+j*2, 0], points_list[i+j*2, 1])
    #         rp_pcc_cmp[1+j] = ax2.scatter(points_list[i+j*2, 0], points_list[i+j*2, 1], points_list[i+j*2, 2], label=f'pos_{j}')
    # else: 
    #     rp_pcc_cmp[0] = ax2.scatter(rp_pcc[0], rp_pcc[1], rp_pcc[2], label='pcc_est', marker='*')
    #     for j in range(4):
    #         rp_pcc_cmp[1+j] = ax2.scatter(points_list[i+j*2, 0], points_list[i+j*2, 1], points_list[i+j*2, 2], label=f'pos_{j}')
    #     ax2.legend()

    # ax2.clear()

    # # rp_pcc_cmp[0] = ax2.scatter(rp_pcc[0], rp_pcc[1], rp_pcc[2], label='rp_pcc_est', marker='*')
    # rp_pcc_cmp[0] = ax2.scatter(0, 0, 0, label='rp_pcc_est', marker='*', c='black')
    # for j in range(4):
    #     rp_pcc_cmp[1+j] = ax2.scatter(points_list[i+j*2, 0] - rp_pcc[0], points_list[i+j*2, 1] - rp_pcc[1], points_list[i+j*2, 2] - rp_pcc[2], label=f'rp_pos_{j+1}')
    
    # for j in range(3):
    #     n_cmp[j] = ax2.plot([0, (neighbors[j, 0] - rp_pcc[0])/10], [0, (neighbors[j, 1] - rp_pcc[1])/10], [0, (neighbors[j, 2] - rp_pcc[2])/10], linestyle='dotted', marker='o', label=f'neib {j+1}')
    # ax2.set_aspect('equal')
    # ax2.legend()

    
    return rp_scatters + n_scatters + sum(rp_cons, []) + sum(n_cons, []) # + rp_pcc_cmp

# Create animation
ani1 = FuncAnimation(fig1, update1, frames=len(points_list) // 7, interval=3000)

# Show plot
plt.show()
