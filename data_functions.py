import os, sys
import tkinter
from tkinter import filedialog
import numpy as np
import seaborn as sns
from math import copysign

tex_plots = False
if tex_plots:
    import matplotlib
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'sans-serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'font.size': 22,
    })
import matplotlib.pyplot as plt
from math import copysign

from kinematics_functions import T_beModule
from point_clouds import generate_square, generate_circle, generate_coil
from draw_functions import draw_robot

from tests_plots_functions import *

def parse_dataset(file_path):
    filename = os.path.basename(os.path.realpath(file_path))
    num_pts = int(filename.split('_')[1])
    traj = np.zeros(shape=(num_pts, 3), dtype=int)
    ref_T = np.zeros(shape=(num_pts, 4, 4))
    pos = np.zeros(shape=(num_pts, 3))
    with open(file_path, 'r') as file:
        i = 0
        line = file.readline()
        while line:
            try:
                if line.startswith('#'):
                    for _ in range(6): line = file.readline()
                    continue
                # print(i)
                traj[i] = [int(num) for num in line.split(', ')]
                # print(file.readline()[2:-2].split())
                # print(file.readline()[2:-2].split())
                ref_T[i, 0, :] = [float(num) for num in file.readline()[2:-2].split()]
                ref_T[i, 1, :] = [float(num) for num in file.readline()[2:-2].split()]
                ref_T[i, 2, :] = [float(num) for num in file.readline()[2:-2].split()]
                ref_T[i, 3, :] = [float(num) for num in file.readline()[2:-3].split()]
                pos[i] = [float(num) for num in file.readline()[1:-2].split(', ')]
                i += 1
                line = file.readline()
            except:
                if i < num_pts:
                    print(f'Failed reading at point #{i+1}, line {i*6 + 1} of txt file')
                    break

    return (traj[:i], ref_T[:i], pos[:i])

def parse_cont_dataset(file_path):
    filename = os.path.basename(os.path.realpath(file_path))
    num_pts = int(filename.split('_')[2])
    traj = np.zeros(shape=(num_pts, 3), dtype=int)
    track_count = np.zeros(shape=(num_pts,), dtype=int)
    # ref_T = np.zeros(shape=(num_pts, 4, 4))
    ref_T = []
    pos = []
    with open(file_path, 'r') as file:
        p_i = 0
        t_i = 0
        # line = file.readline()
        while (line := file.readline()):
            try:
                if line.startswith('['):
                    ref_T.append(np.empty((4,4)))
                    ref_T[-1][0, :] = [float(num) for num in line[2:-2].split()]
                    ref_T[-1][1, :] = [float(num) for num in file.readline()[2:-2].split()]
                    ref_T[-1][2, :] = [float(num) for num in file.readline()[2:-2].split()]
                    ref_T[-1][3, :] = [float(num) for num in file.readline()[2:-3].split()]
                    pos.append(np.array([float(num) for num in file.readline()[1:-2].split(', ')]))
                    p_i += 1
                    # if p_i < 20:
                    #     print(ref_T[-1])
                    #     print(pos[-1])
                elif line.startswith('--'):
                    track_count[t_i] = int(line[:-1].strip('-'))
                else:
                    traj[t_i] = [int(num) for num in line.split(', ')]
                    # if p_i < 20:
                    #     print(track_count[t_i], sum(track_count[:t_i+1]), traj[t_i])
                    t_i += 1
            except Exception as e:
                if t_i < num_pts:
                    print(f'Failed reading at point #{p_i+1}, line {p_i*6 + 1} of txt file')
                    print(line)
                    print(e)
                    print(ref_T[-1])
                    break

    print('\n')
    return traj[:t_i], np.array(ref_T), np.array(pos), track_count[:t_i]


def polaris2base(traj, ref_T, pos):

    global jtheta2len

    ref_p = ref_T[:,0:3,3]
    ref_R = ref_T[:,0:3,0:3]

    dx = -18; dy = -104; dz = -83; alpha = 45 * np.pi / 180; beta = 0

    pos_ref = np.array([ref_R[i].T @ (pos[i] - ref_p[i]) for i in range(len(pos))]) # position relative to reference tool
    # pos_ref = np.einsum('ijk,ik->ij', ref_R.T, pos - ref_p) # vectorized version (for larger datasets)

    base_R = np.array([[1, 0, 0],
                       [0, np.cos(alpha), -np.sin(alpha)],
                       [0, np.sin(alpha), np.cos(alpha)],
                      ])

    # left-handed coordinate system
    pos_base = (pos_ref - np.array([[dx, dy, dz]])) @ base_R @ np.array([[0,0,1],
                                                                         [-1,0,0],
                                                                         [0,1,0]])

    # pos_base = (pos_ref - np.array([[dx, dy, dz]])) @ base_R @ np.array([[0,0,1],
    #                                                                      [1,0,0],
    #                                                                      [0,1,0]])

    # if traj[0][0] > 500:
    jtheta2len = lambda p: (p - 4488.62157) / -26.03760 # writeMicroseconds, polaris readings
    #     # jtheta2len = lambda p: (p - 4552.15581) / -25.86452 # writeMicroseconds, manual readings
    # else: 
    #     jtheta2len = lambda p: (p - 383.89841) / -2.53968 # polaris readings
    #     # jtheta2len = lambda p: (p - 384.1) / -2.48336 # manual readings

    traj_ = jtheta2len(traj)
    # if beta == 0:
    est_pos_base = np.array([T_beModule(p, [], 0, 0)[0:3,3] for p in traj_])
    # else: 
    #     R_pcc_to_base = np.array([[np.cos(beta), -np.sin(beta), 0],
    #                               [np.sin(beta), np.cos(beta), 0],
    #                               [0, 0, 1]])
    #     est_pos_base = np.array([T_beModule(p, [], 0, 0)[0:3,3] @ R_pcc_to_base for p in traj_])

    return traj_, pos_base, est_pos_base


def auto_vert_test_analysis(filenames):
    for i, filename in enumerate(filenames):
        if 'vert' not in filename[0]: print("Not Vertical Test")

        traj, ref_T, pos = parse_dataset(filename[0])

        traj_, pos_base, est_pos_base = polaris2base(traj, ref_T, pos)

        # rel_err_norm = np.linalg.norm(pos_base - est_pos_base, axis=1)  / np.linalg.norm(est_pos_base, axis=1)
        abs_err_norm = np.linalg.norm(pos_base - est_pos_base, axis=1)
        rel_err_norm = abs_err_norm  / 125.5
        print('\nDataset:', filename[0])
        print('Mean Relative error:', np.mean(rel_err_norm) * 100, '%')
        print('Max Relative error:', np.max(np.abs(rel_err_norm)) * 100, '%', f'(point {np.argmax(np.abs(rel_err_norm))})')
        print('Min Relative error:', np.min(np.abs(rel_err_norm)) * 100, '%')

        # plot_vert_test(20, pos_base, est_pos_base)
        plot_vert_test(0, pos_base, est_pos_base)
        # plot_rel_err(pos_base, est_pos_base, rel_err_norm, np.mean(traj_, axis=1))
        # plot_abs_err(pos_base, est_pos_base, abs_err_norm, np.mean(traj_, axis=1), xyz_flag=False, d3d_flag=False)



def auto_curv_test_analysis_err(filenames):
    max_fixed_pos_len = max([len(fixed_pos) for _, fixed_pos in filenames])
    fig1, ax1 = plt.subplots(max_fixed_pos_len, 3, figsize=(16,12))
    fig1.suptitle("Absolute Error (mm) as a function of lengths of non-fixed flexible rods (mm)")
    fig2, ax2 = plt.subplots(max_fixed_pos_len, 3, figsize=(16,12))
    fig2.suptitle("Absolute Error (mm) as a function of length of non-fixed flexible rods (mm)")
    not_first = False
    for i, filename in enumerate(filenames):
        if 'curv2' not in filename[0]:
            print("Not Curv Test, skipping")
            continue
        traj, ref_T, pos = parse_dataset(filename[0])

        fixed_pos = filename[1]

        traj_, pos_base, est_pos_base = polaris2base(traj, ref_T, pos)

        rel_err_norm = np.linalg.norm(pos_base - est_pos_base, axis=1)  / np.linalg.norm(est_pos_base, axis=1)
        print('\nDataset:', filename[0])
        print('Mean Relative error:', np.mean(rel_err_norm) * 100, '%')
        print('Max Relative error:', np.max(np.abs(rel_err_norm)) * 100, '%', f'(point {np.argmax(np.abs(rel_err_norm))})')
        print('Min Relative error:', np.min(np.abs(rel_err_norm)) * 100, '%')

        plot_curv_test_err2(fixed_pos, 1, traj, traj_, pos_base, est_pos_base, ax1, plt_label=f"test {i+1}", not_first=not_first)
        plot_curv_test_err2(fixed_pos, 2, traj, traj_, pos_base, est_pos_base, ax2, plt_label=f"test {i+1}", not_first=not_first)
        not_first = True

        # plot_basic(pos_base, est_pos_base, rel_err_norm, np.mean(traj_, axis=1))

def auto_curv_test_analysis_2d(filenames):
    fig1, ax1 = plt.subplots(3, 2, figsize=(10,10))
    # fig1.suptitle("Trajectory in xy, rotated to align with x axis") # traj1
    fig1.suptitle("Trajectory in x and z, with points rotated to align with y") # traj2
    # fig1.suptitle("Trajectory in x, y and z, with points rotated to align with y") # traj2
    # fig2, ax2 = plt.subplots(3, 3, figsize=(8,6))
    # fig2.suptitle("Trajectory in xz")
    # fig2.suptitle("Absolute Error in z relative to cable length")
    est_flag = True
    for i, filename in enumerate(filenames):
        if 'curv2' not in filename[0]:
            print("Not Curv Test, skipping")
            continue
        traj, ref_T, pos = parse_dataset(filename[0])

        fixed_pos = filename[1]

        traj_, pos_base, est_pos_base = polaris2base(traj, ref_T, pos)

        rel_err_norm = np.linalg.norm(pos_base - est_pos_base, axis=1)  / np.linalg.norm(est_pos_base, axis=1)
        print('\nDataset:', filename[0])
        print('Mean Relative error:', np.mean(rel_err_norm) * 100, '%')
        print('Max Relative error:', np.max(np.abs(rel_err_norm)) * 100, '%', f'(point {np.argmax(np.abs(rel_err_norm))})')
        print('Min Relative error:', np.min(np.abs(rel_err_norm)) * 100, '%')


        # plot_curv_test_2d_traj(fixed_pos, 2, traj, traj_, pos_base, est_pos_base, est_flag, ax1, ax2, plt_label=f"test {i+1}")
        plot_curv_test_2d_traj2(fixed_pos, 2, traj, traj_, pos_base, est_pos_base, est_flag, ax1, plt_label=f"test {i+1}")
        est_flag=False

        # plot_basic(pos_base, est_pos_base, rel_err_norm, np.mean(traj_, axis=1))

def auto_curv_test_analysis_3d(filenames):
    # fig1, ax1 = plt.subplots(2, 3, figsize=(15,12), subplot_kw={'projection': '3d', 'aspect': 'equal'})
    # fig2, ax2 = plt.subplots(2, 3, figsize=(15,12), subplot_kw={'projection': '3d', 'aspect': 'equal'})
    # for i in range(3):
    #     ax1[0,i].view_init(elev=0, azim=90)
    #     ax1[0,i].set_zlim(0, 155)
    #     ax1[0,i].set_yticks([])

    #     ax1[1,i].view_init(elev=90, azim=0)
    #     ax1[1,i].set_zticks([])
    #     # ax1[1,i].set_zlim(75, 155)
    #     ax1[1,i].set_ylim(-25, 25)
    #     ax1[1,i].set_yticks([-20, 0, 20])
    #     # ax1[1,i].set_yticklabels([-20, 0, 20], va='top')

    #     ax1[1,i].tick_params('x', pad=30)
    #     ax1[1,i].tick_params('y', pad=5)
    #     ax1[1,i].tick_params('z', pad=10)

    #     ax2[0,i].view_init(elev=0, azim=90)
    #     ax2[0,i].set_zlim(0, 155)
    #     ax2[0,i].set_yticks([])

    #     ax2[1,i].view_init(elev=90, azim=0)
    #     ax2[1,i].set_zticks([])
    #     # ax2[1,i].set_zlim(75, 155)
    #     ax2[1,i].set_ylim(-25, 25)
    #     ax2[1,i].set_yticks([-20, 0, 20])
    #     # ax2[1,i].set_yticklabels([-20, 0, 20], va='top')

    #     ax2[1,i].tick_params('x', pad=30)
    #     ax2[1,i].tick_params('y', pad=5)
    #     ax2[1,i].tick_params('z', pad=10)

        # ax1[0,i].view_init(elev=45, azim=30)
        # ax1[1,i].view_init(elev=45, azim=30)
        # ax1[2,i].view_init(elev=45, azim=30)
    # ax1 = fig1.add_subplot(331, projection='3d', )
    est_flag = True
    for i, filename in enumerate(filenames):
        if 'curv2' not in filename[0]:
            print("Not Curv Test, skipping")
            continue
        traj, ref_T, pos = parse_dataset(filename[0])

        fixed_pos = filename[1]

        traj_, pos_base, est_pos_base = polaris2base(traj, ref_T, pos)

        rel_err_norm = np.linalg.norm(pos_base - est_pos_base, axis=1)  / np.linalg.norm(est_pos_base, axis=1)
        print('\nDataset:', filename[0])
        print('Mean Relative error:', np.mean(rel_err_norm) * 100, '%')
        print('Max Relative error:', np.max(np.abs(rel_err_norm)) * 100, '%', f'(point {np.argmax(np.abs(rel_err_norm))})')
        print('Min Relative error:', np.min(np.abs(rel_err_norm)) * 100, '%')

        plot_curv_test_3d(fixed_pos, 1, traj, pos_base, est_pos_base, est_flag, i, plt_label=f"")
        # fig1.suptitle('Test with 1 Cable Fixed')
        # plot_curv_test_3d(fixed_pos, 2, traj, pos_base, est_pos_base, est_flag, i, plt_label=f"test {i+1}")
        # fig2.suptitle('Test with 2 Cables Fixed')

        est_flag = False

def auto_basic_analysis(filenames, concat=True):
    total_datapoints = 0
    for i, filename in enumerate(filenames):
        traj, ref_T, pos = parse_dataset(filename[0])

        # fixed_pos = filename[1]

        traj_, pos_base, est_pos_base = polaris2base(traj, ref_T, pos)
        pos_base[:,:2] = pos_base[:,:2] - np.mean(pos_base[:,:2], axis=0)
        abs_err_norm = np.linalg.norm(pos_base - est_pos_base, axis=1)
        # rel_err_norm =  abs_err_norm / np.linalg.norm(est_pos_base, axis=1)
        rel_err_norm =  abs_err_norm / 125.5
        print('\nDataset:', filename[0])
        # print('Mean Relative error:', np.mean(rel_err_norm) * 100, '%')
        # print('Max Relative error:', np.max(np.abs(rel_err_norm)) * 100, '%', f'(point {np.argmax(np.abs(rel_err_norm))})')
        # print('Min Relative error:', np.min(np.abs(rel_err_norm)) * 100, '%')
        print('Mean Absolute error:', np.mean(abs_err_norm), 'mm')
        print('std Absolute error:', np.std(abs_err_norm), 'mm')
        print('Max Absolute error:', np.max(np.abs(abs_err_norm)), 'mm', f'(point {np.argmax(np.abs(abs_err_norm))})')
        print('Min Absolute error:', np.min(np.abs(abs_err_norm)), 'mm')
        # print('Err > mean: ', len(pos_base[abs_err_norm > np.mean(abs_err_norm)]))

        if concat and i == 0:
            concat_pos_base = pos_base.copy()
            concat_est_pos_base = est_pos_base.copy()
            concat_traj_ = traj_.copy()
            concat_abs_err_norm = abs_err_norm.copy()
            concat_rel_err_norm = rel_err_norm.copy()
        elif concat:
            concat_pos_base = np.concatenate((pos_base, concat_pos_base), axis=0)
            concat_est_pos_base = np.concatenate((est_pos_base, concat_est_pos_base), axis=0)
            concat_traj_ = np.concatenate((traj_, concat_traj_), axis=0)
            concat_abs_err_norm = np.concatenate((abs_err_norm, concat_abs_err_norm), axis=0)
            concat_rel_err_norm = np.concatenate((rel_err_norm, concat_rel_err_norm), axis=0)
        else:
            # plot_rel_err(pos_base, est_pos_base, rel_err_norm, np.mean(traj_, axis=1), xyz_flag=False, d3d_flag=True)
            plot_abs_err(pos_base, est_pos_base, abs_err_norm, np.mean(traj_, axis=1), xyz_flag=False, d3d_flag=True)
            # plot_vert_test(0, pos_base, est_pos_base)
        # plot_grid(traj_)
        total_datapoints += len(pos_base)

    if concat:
        print("Concat MSE:", np.mean(np.linalg.norm(concat_pos_base - concat_est_pos_base, axis=1)))
        print("Mean pos:", np.mean(concat_pos_base, axis=0))
        plot_rel_err(concat_pos_base, concat_est_pos_base, concat_rel_err_norm, np.mean(concat_traj_, axis=1), xyz_flag=False, d3d_flag=True)
        plot_abs_err(concat_pos_base, concat_est_pos_base, concat_abs_err_norm, np.mean(concat_traj_, axis=1), xyz_flag=False, d3d_flag=True)
        # plot_vert_test(0, concat_pos_base, concat_est_pos_base)

    print("total", total_datapoints)

def repeat_analysis(filenames):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    concat_pos_base = np.empty((len(filenames), 10, 3))
    concat_err = np.empty((len(filenames), 10, 3))
    for i, filename in enumerate(filenames):
        traj, ref_T, pos = parse_dataset(filename[0])            

        if len(pos) != 10:
            continue
        traj_, pos_base, est_pos_base = polaris2base(traj, ref_T, pos)
        abs_err_norm = np.linalg.norm(pos_base - est_pos_base, axis=1)
        # rel_err_norm =  abs_err_norm / np.linalg.norm(est_pos_base, axis=1)
        print('\nDataset:', filename[0])
        # print('Mean Relative error:', np.mean(rel_err_norm) * 100, '%')
        # print('Max Relative error:', np.max(np.abs(rel_err_norm)) * 100, '%', f'(point {np.argmax(np.abs(rel_err_norm))})')
        # print('Min Relative error:', np.min(np.abs(rel_err_norm)) * 100, '%')
        print('Mean Absolute error to PCC:', np.mean(abs_err_norm), 'mm')
        print('Max Absolute error to PCC:', np.max(np.abs(abs_err_norm)), 'mm', f'(point {np.argmax(np.abs(abs_err_norm))})')
        print('Min Absolute error to PCC:', np.min(np.abs(abs_err_norm)), 'mm')

        if i > 0:
            abs_err_check = np.linalg.norm(pos_base - prev_pos_base, axis=1) # pyright: ignore

            print(f'Mean Absolute error to previous check: {np.mean(abs_err_check):.2f}, mm')
            print(f'Max Absolute error to previous check: {np.max(np.abs(abs_err_check)):2f}, mm point {np.argmax(np.abs(abs_err_check)):.2f})')
            print(f'Min Absolute error to previous check: {np.min(np.abs(abs_err_check)):.2f} mm')

        # plot_abs_err(pos_base, est_pos_base, abs_err_norm, np.mean(traj_, axis=1), annotate_flag=True, xyz_flag=False, d3d_flag=False)

        # if i == 0:
        #     plot_3d(pos_base, est_pos_base, ax, idx=f'{i+1}', pcc_flag=True)
        # else:
        #     plot_3d(pos_base, est_pos_base, ax, idx=f'{i+1}', pcc_flag=False)
        prev_pos_base = pos_base
        concat_pos_base[i, :, :] = pos_base
        concat_err[i, :, :] = pos_base - est_pos_base

    for i, set_p in enumerate(concat_pos_base):
        if i == 0:
            plot_3d(set_p, est_pos_base, ax, idx=f'{i+1}', pcc_flag=True)
        else:
            plot_3d(set_p, est_pos_base, ax, idx=f'{i+1}', pcc_flag=False)

    ax.legend()
    draw_robot(ax, alpha_mult=0.4)
    ax.set_title("Repeatability Analysis")
    ax.set_aspect('equal')

    
    print("Std. Dev. of the Position per axis:", np.mean(np.std(concat_pos_base, axis=0), axis=0), 'mm')
    print("Std. Dev. of the Norm of the Position:", np.mean(np.std(np.linalg.norm(concat_pos_base, axis=2), axis=0)), 'mm')
    print("Std. Dev. of the CC Error per axis:", np.mean(np.std(concat_err, axis=1), axis=0), 'mm')
    print("Std. Dev. of the CC Error of the Position:", np.mean(np.std(np.linalg.norm(concat_err, axis=2), axis=0)), 'mm')

    # print(concat_pos_base[:2])
    # print(np.std(concat_pos_base, axis=1))
