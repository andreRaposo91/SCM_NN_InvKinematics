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
    })
import matplotlib.pyplot as plt
from math import copysign
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'sans-serif'

from kinematics_functions import T_beModule, jtheta2len
from point_clouds import generate_square, generate_circle, generate_coil
from draw_functions import draw_robot

def plot_curv_test_3d(fixed_pos, num_fixed, traj, pos_base, est_pos_base, est_flag, ax, plt_label='test'):
    mask_curv_test = [[False] * len(traj)] * 9

    traj_ = jtheta2len(traj)

    for i, fix_pos in enumerate(fixed_pos):
        for j in range(3):
            idx = [0,1,2]
            idx.remove(j)
            if num_fixed == 1:
                mask_curv_test[i*3+j] = ((traj[:,j] == fix_pos) & (traj[:,idx[0]] == traj[:,idx[1]])) # 1 cable fixed, 2 moving
            elif num_fixed == 2:
                mask_curv_test[i*3+j] = (traj[:,idx[0]] == fix_pos) & (traj[:,idx[1]] == fix_pos) # 2 cables fixed, 1 moving

    ang = [0, np.radians(-120), np.radians(-240)]
    if num_fixed == 1: servos = [1,2,0]; s=''
    if num_fixed == 2: servos = [0,1,2]; s='s'

    figs = [plt.figure(figsize=(11,5)) for _ in range(len(fixed_pos))]
    xz_axs = [fig.add_subplot(121, projection='3d') for fig in figs]
    xy_axs = [fig.add_subplot(122, projection='3d') for fig in figs]
    for i in range(3):
        # if i == 0:
        #     test_fig, test_ax = plt.subplots(1, 2, figsize=(10,4))
        #     test_ax[0].plot(traj_[mask_curv_test[i]], label=['Cable 1', 'Cable 2', 'Cable 3'])
        #     test_ax[0].legend()
        #     test_ax[1].plot(est_pos_base[mask_curv_test[i], 0], est_pos_base[mask_curv_test[i], 2])
        for j in range(1):

            # print(i, j)

            pos_base_rot = np.array([np.array([[np.cos(ang[j]), -np.sin(ang[j]), 0],
                                        [np.sin(ang[j]), np.cos(ang[j]), 0],
                                        [0, 0, 1]]) @ p for p in pos_base[mask_curv_test[i*3+j]]])
            est_pos_base_rot = np.array([np.array([[np.cos(ang[j]), -np.sin(ang[j]), 0],
                                        [np.sin(ang[j]), np.cos(ang[j]), 0],
                                        [0, 0, 1]]) @ p for p in est_pos_base[mask_curv_test[i*3+j]]])

            pos_base_rot_neg = pos_base_rot[np.insert(np.where(np.diff(traj_[mask_curv_test[i*3+j], servos[0]]) < 0)[0]+1, 0, 0)]
            pos_base_rot_pos = np.insert(pos_base_rot[np.where(np.diff(traj_[mask_curv_test[i*3+j], servos[0]]) >= 0)[0]+1], 0, pos_base_rot_neg[-1], axis=0)

            if est_flag:
                xz_axs[i].plot(*zip(*est_pos_base_rot), marker="*", linestyle=":", label="CC Estimate " + plt_label)
                xy_axs[i].plot(*zip(*est_pos_base_rot), marker="*", linestyle=":", label="CC Estimate " + plt_label)
                # xz_axs[i].plot(*zip(*est_pos_base_rot), marker="o", markersize=4, linestyle=":")
                # xy_axs[i].plot(*zip(*est_pos_base_rot), marker="o", markersize=4, linestyle=":")

            # xz_axs[i].plot(*zip(*pos_base_rot_neg), marker="x", label='Moving Down', markersize=5, color='r', alpha=0.7)
            # xz_axs[i].plot(*zip(*pos_base_rot_pos), marker="x", label='Moving Up', markersize=5, color='g', alpha=0.7)
            # xy_axs[i].plot(*zip(*pos_base_rot_neg), marker="x", label='Moving Down', markersize=5, color='r', alpha=0.7)
            # xy_axs[i].plot(*zip(*pos_base_rot_pos), marker="x", label='Moving Up', markersize=5, color='g', alpha=0.7)
            # # ax[i,j].plot(*zip(*pos_base_rot), marker="o", label=plt_label, markersize=5)
            # xz_axs[i].scatter(*pos_base_rot[0], marker="s", label="Start", color="g")
            # xy_axs[i].scatter(*pos_base_rot[0], marker="s", label="Start", color="g")
            # # ax[i,j].scatter(*pos_base_rot[0], marker="s", label="Start", color="g")
            # xz_axs[i].scatter(*pos_base_rot[-1], marker="s", label="Finish", color="r")
            # xy_axs[i].scatter(*pos_base_rot[-1], marker="s", label="Finish", color="r")
            # ax[i,j].scatter(*pos_base_rot[-1], marker="s", label="Finish", color="r")
            draw_robot(xz_axs[i], alpha_mult=0.4)
            draw_robot(xy_axs[i], alpha_mult=0.3)
            # draw_robot(ax[i,j], alpha_mult=0.6)
            figs[i].suptitle(f"Length of Fixed Flexible Rod{s} : {jtheta2len(fixed_pos[i]):.2f} mm")
            xz_axs[i].set_title("Side View")
            xy_axs[i].set_title("Top View")
            # ax[i,j].set_title(f"Fixed Length of Cable {j}: {jtheta2len(fixed_pos[i]):.2f} mm")

                # ax[i,j].plot(*zip(*est_pos_base_rot), marker="*", linestyle=":", label="est " + plt_label)
            xz_axs[i].set_xlabel('x [mm]')
            xz_axs[i].set_zlabel('z [mm]')
            xy_axs[i].set_xlabel(r'x [mm]', labelpad=60)
            xy_axs[i].set_ylabel('y [mm]')
            # xy_axs[i].set_zlabel('z [mm]', labelpad=20)
            # xy_axs[i].set_xticklabels(map(int, xy_axs[i].get_xticks()), va='bottom')

            # if i == 0:
            #     xz_axs[i].legend(loc='upper left', bbox_to_anchor=(-0.4, 0.6))
                # xy_axs[i].legend(loc='upper left', bbox_to_anchor=(-0.4, 0.6))
            xz_axs[i].legend(loc='upper left', bbox_to_anchor=(-0.3, 0.6))

            xz_axs[i].view_init(elev=0, azim=90)
            xz_axs[i].set_zlim(0, 155)
            xz_axs[i].set_yticks([])

            xy_axs[i].view_init(elev=90, azim=0)
            xy_axs[i].set_zticks([])
            # xy_axs[i].set_zlim(75, 155)
            xy_axs[i].set_ylim(-25, 25)
            xy_axs[i].set_yticks([-20, 0, 20])
            # xy_axs[i].set_yticklabels([-20, 0, 20], va='top')

            xy_axs[i].tick_params('x', pad=30)
            xy_axs[i].tick_params('y', pad=5)
            xy_axs[i].tick_params('z', pad=10)

            # ax[i,j].legend()


def plot_curv_test_err2(fixed_pos, num_fixed, traj, traj_, pos_base, est_pos_base, ax, plt_label='test', not_first=False):
    mask_curv_test = [[False] * len(traj)] * len(fixed_pos) * 3
        
    max_mask_len = 0
    for i, fix_pos in enumerate(fixed_pos):
        for j in range(3):
            idx = [0,1,2]
            idx.remove(j)
            if num_fixed == 1:
                mask_curv_test[i*3+j] = ((traj[:,j] == fix_pos) & (traj[:,idx[0]] == traj[:,idx[1]])) # 1 cable fixed, 2 moving
            elif num_fixed == 2:
                mask_curv_test[i*3+j] = (traj[:,idx[0]] == fix_pos) & (traj[:,idx[1]] == fix_pos) # 2 cables fixed, 1 moving
            if (m := len(mask_curv_test[i*3+j])) > max_mask_len:
                max_mask_len = m


    # print((min_len, max_len))

    # _, unique_idx = np.unique(traj, return_index=True, axis=0)

    # unique_mask = np.array([False] * len(traj))

    # print(unique_mask.shape)

    # unique_mask[unique_idx] = True

    # mask_curv_test = mask_curv_test & unique_mask

    if num_fixed == 1: servos = [1, 2, 0]
    if num_fixed == 2: servos = [0, 1, 2]

    if num_fixed == 1:
        ang0 = 0
        ang1 = np.radians(-120)
        ang2 = np.radians(-240)
    else:
        ang0 = 0
        ang1 = np.radians(-120)
        ang2 = np.radians(-240)

    axis_labels = ['Error in x [mm]', 'Error in y [mm]', 'Error in z [mm]']

    # test_fig, test_ax = plt.subplots(1, len(fixed_pos), figsize=(14,4))
    # test_fig.tight_layout(pad=3.5, w_pad=7.5)
    test_ax = []
    # test_fig = plt.figure()
    # test_ax = test_fig.add_subplot(111)
    test_plur = 's' if num_fixed == 2 else ''
    # test_fig.suptitle(f'Cable Lengths during Test w/ {num_fixed} fixed cable{test_plur}')
    test_colors = ('r', 'b', 'g')
    test_labels = ('Moving FR', 'Fixed FR') if num_fixed == 1 else ('Moving FR', 'Fixed FR')

    # abs_err_fig, abs_err_axs = plt.subplots(1, len(fixed_pos), figsize=(14,4))
    # abs_err_fig.tight_layout(pad=3.5, w_pad=7.5)
    abs_err_axs = []

    tick_num = 7
    tick_half = int(np.ceil(tick_num/2))
    for i in range(len(fixed_pos)):
        _, abs_err_axs_tmp = plt.subplots(1, 1)
        abs_err_axs.append(abs_err_axs_tmp)
        _, test_ax_tmp = plt.subplots(1, 1)
        test_ax.append(test_ax_tmp)
        # test_fig.suptitle(f'Cable Lengths during Test w/ {num_fixed} fixed cable{test_plur}')

        mask0 = mask_curv_test[i*3]

        max_len = np.max(traj_[mask0])
        min_len = np.min(traj_[mask0])
        pos_base_rot0 = np.array([np.array([[np.cos(ang0), -np.sin(ang0), 0],
                        [np.sin(ang0), np.cos(ang0), 0],
                        [0, 0, 1]]) @ p for p in pos_base[mask0]])
        est_pos_base_rot0 = np.array([np.array([[np.cos(ang0), -np.sin(ang0), 0],
                        [np.sin(ang0), np.cos(ang0), 0],
                        [0, 0, 1]]) @ p for p in est_pos_base[mask0]])
        # x_lim = (-len(pos_base_rot0)//2, len(pos_base_rot0)//2 + len(pos_base_rot0)%2)

        neg_sec = traj_[mask0, servos[0]][np.insert(np.where(np.diff(traj_[mask0, servos[0]]) < 0)[0]+1, 0, 0)]
        pos_sec = traj_[mask0, servos[0]][np.where(np.diff(traj_[mask0, servos[0]]) >= 0)[0]+1]
        # neg_plot_len0 = np.interp(neg_sec, [min_len, max_len], [0, -1])
        # pos_plot_len0 = np.interp(pos_sec, [min_len, max_len], [0, 1])
        plot_len0 = np.concatenate((np.interp(neg_sec, [min_len, max_len], [0, -1]), np.interp(pos_sec, [min_len, max_len], [0, 1])))
        if num_fixed == 1:
            assert traj_[mask0, servos[-1]][0] == traj_[mask0, servos[-1]][1]
            rest_len = 1.5 * (125.5 - traj_[mask0, servos[-1]][0]/3)
        else:
            assert traj_[mask0, servos[-1]][0] == traj_[mask0, servos[-1]][1]
            rest_len = 3 * (125.5 - traj_[mask0, servos[-1]][0]*2/3)
        rest_len = [126, 125.5, 124][i]  # override with measured values
        vert_lines = (np.interp(rest_len, [min_len, max_len], [0, -1]), np.interp(rest_len, [min_len, max_len], [0, 1]))
        # print(traj_[mask0, servos[-1]][0], rest_len)
        # print(neg_sec, len(neg_sec), neg_plot_len0)
        # print(pos_sec, len(pos_sec), pos_plot_len0)
        assert(len(plot_len0) == len(pos_base_rot0))
        # print(np.concatenate((np.insert(np.where(np.diff(traj_[mask0, servos[0]]) < 0)[0]+1, 0, 0),
        #                       np.where(np.diff(traj_[mask0, servos[0]]) >= 0)[0]+1)))
        # print(len(neg_plot_len0), '+', len(pos_plot_len0), '==', len(pos_base_rot0))
        # plot_len0 = list(range(len(pos_base_rot0)))
        # lin_fit = LinearRegression()
        # lin_fit.fit(traj_[mask_curv_test[i*3]], plot_len0)

        # fixed_pos_len = (np.interp(jtheta2len(fixed_pos[i]), [min_len, max_len], [0, -1]), np.interp(jtheta2len(fixed_pos[i]), [min_len, max_len], [0, 1]))
        fixed_pos_len = (np.interp(rest_len, [min_len, max_len], [0, -1]), np.interp(rest_len, [min_len, max_len], [0, 1]))

        # print(jtheta2len(fixed_pos[i]), fixed_pos_len)

        mask1 = mask_curv_test[i*3+1]
        pos_base_rot1 = np.array([np.array([[np.cos(ang1), -np.sin(ang1), 0],
                        [np.sin(ang1), np.cos(ang1), 0],
                        [0, 0, 1]]) @ p for p in pos_base[mask1]])
        est_pos_base_rot1 = np.array([np.array([[np.cos(ang1), -np.sin(ang1), 0],
                        [np.sin(ang1), np.cos(ang1), 0],
                        [0, 0, 1]]) @ p for p in est_pos_base[mask1]])
        # plot_len1 = range(len(pos_base_rot1))
        # plot_len1 = lin_fit.predict(traj_[mask_curv_test[i*3+1]])
        neg_sec = traj_[mask1, servos[1]][np.insert(np.where(np.diff(traj_[mask1, servos[1]]) < 0)[0]+1, 0, 0)]
        pos_sec = traj_[mask1, servos[1]][np.where(np.diff(traj_[mask1, servos[1]]) >= 0)[0]+1]
        plot_len1 = np.concatenate((np.interp(neg_sec, [min_len, max_len], [0, -1]), np.interp(pos_sec, [min_len, max_len], [0, 1])))


        mask2 = mask_curv_test[i*3+2]
        pos_base_rot2 = np.array([np.array([[np.cos(ang2), -np.sin(ang2), 0],
                        [np.sin(ang2), np.cos(ang2), 0],
                        [0, 0, 1]]) @ p for p in pos_base[mask2]])
        est_pos_base_rot2 = np.array([np.array([[np.cos(ang2), -np.sin(ang2), 0],
                        [np.sin(ang2), np.cos(ang2), 0],
                        [0, 0, 1]]) @ p for p in est_pos_base[mask2]])
        # plot_len2 = range(len(pos_base_rot2))
        # plot_len2 = lin_fit.predict(traj_[mask_curv_test[i*3+2]])
        neg_sec = traj_[mask2, servos[2]][np.insert(np.where(np.diff(traj_[mask2, servos[2]]) < 0)[0]+1, 0, 0)]
        pos_sec = traj_[mask2, servos[2]][np.where(np.diff(traj_[mask2, servos[2]]) >= 0)[0]+1]
        plot_len2 = np.concatenate((np.interp(neg_sec, [min_len, max_len], [0, -1]), np.interp(pos_sec, [min_len, max_len], [0, 1])))

        # print(pos_base_rot0.shape, pos_base_rot1.shape, pos_base_rot2.shape)

        # print(np.where(np.diff(traj_[mask0,servos[0]][pos_base_rot0.shape[0]//2-2:pos_base_rot0.shape[0]//2+3]) > 0),
        #       np.where(np.diff(traj_[mask1,servos[1]][pos_base_rot0.shape[0]//2-2:pos_base_rot0.shape[0]//2+3]) > 0),
        #       np.where(np.diff(traj_[mask2,servos[2]][pos_base_rot0.shape[0]//2-2:pos_base_rot0.shape[0]//2+3]) > 0))


        # x = i
        # test_servos.remove((x+1)%3)
        # test_ax[x].set_title(f'Cable Lengths during Test w/ {num_fixed} fixed cable{test_plur}')
        # test_labels = ('Moving Cables', f'Fixed Cable - {traj_[mask_curv_test[x*3]][0, test_servos[1]]:.2f}mm') if num_fixed == 1 else ('Moving Cable', f'Fixed Cables - {traj_[mask_curv_test[x*3]][0, test_servos[1]]:.2f}mm')
        test_ax[i].set_ylabel('Cable Length [mm]')
        test_ax[i].set_xlabel('Sample Points')
        test_ax[i].set_ylim((80,150))
        # test_ax[i].hlines(rest_len, 0, len(traj_[mask0]), label=f'{test_labels[0]} Length for \nSpring to be in Resting Length', linestyle='dotted')
        # test_ax[i].hlines(rest_len, 0, len(traj_[mask0]), label=f'Spring Resting Length Position', linestyle='dotted')
        test_ax[i].plot(traj_[mask0][:,servos[0]], color='r', label=test_labels[0], marker='o', markersize=4)
        test_ax[i].plot(traj_[mask0][:,servos[-1]], color='b', label=test_labels[1], marker='o', markersize=4)
        # test_ax[i].legend()
        test_ax[i].set_title(f'Length of {test_labels[1]}: {round(traj_[mask0, servos[-1]][0])} mm')
        # test_ax[x].set_xticklabels(map(int, test_ax[x].get_xticks()))
        test_ax[i].set_xticks(range(0, len(traj_[mask0][:,servos[-1]]), 2))
        test_ax[0].legend(loc='lower left', bbox_to_anchor=(.6, 0))

        # result_fig, result_ax = plt.subplots(1, 3, figsize=(12,4))
        # result_ax[0].plot(est_pos_base_rot0[:, 0], est_pos_base_rot0[:, 2])
        # result_ax[1].plot(est_pos_base_rot1[:, 0], est_pos_base_rot1[:, 2])
        # result_ax[2].plot(est_pos_base_rot2[:, 0], est_pos_base_rot2[:, 2])
        # s1 = result_ax[0].plot(pos_base_rot0[:,0], pos_base_rot0[:,2], label="Along Cable 1") #  marker='o', markersize=5
        # s1 = result_ax[1].plot(pos_base_rot1[:,0], pos_base_rot1[:,2], label="Along Servo 2") #  marker='o', markersize=5
        # s1 = result_ax[2].plot(pos_base_rot2[:,0], pos_base_rot2[:,2], label="Along Servo 3") #  marker='o', markersize=5
        # print()

        abs_err_curv = [np.linalg.norm(pos_base_rot0 - est_pos_base_rot0, axis=1),
            np.linalg.norm(pos_base_rot1 - est_pos_base_rot1, axis=1),
            np.linalg.norm(pos_base_rot2 - est_pos_base_rot2, axis=1)]

        abs_err_axs[i].plot(plot_len0, abs_err_curv[0], label="Along Servo 1")
        abs_err_axs[i].plot(plot_len1, abs_err_curv[1], label="Along Servo 2")
        abs_err_axs[i].plot(plot_len2, abs_err_curv[2], label="Along Servo 3")
        abs_err_axs[i].set_xticks(np.linspace(-1, 1, tick_num), labels=[f"{num:.1f}" for num in np.concatenate((np.linspace(max_len, min_len, tick_half), np.linspace(min_len, max_len, tick_half)[1:]))])
        # abs_err_axs[i].set_ylim([0, max(14, 1.1*np.max(np.concatenate(abs_err_curv)))])
        abs_err_axs[i].set_ylim([0, 14])
        if vert_lines[1] < 1:
                # print(vert_lines)
            # abs_err_axs[i].vlines(vert_lines, [abs_err_axs[i].get_ylim()[0]]*2, [abs_err_axs[i].get_ylim()[1]]*2, color='r', label='Spring Resting Length\nPosition', linestyle='dotted')
            abs_err_axs[i].vlines(fixed_pos_len, [abs_err_axs[i].get_ylim()[0]]*2, [abs_err_axs[i].get_ylim()[1]]*2, color='k', label='Fixed FR Length', linestyle='dotted')

        abs_err_axs[i].set_xlabel('Moving Cable Length [mm]')
        abs_err_axs[i].set_ylabel('Absolute Error [mm]')
        if num_fixed == 1: abs_err_axs[i].set_title(f"Length of Fixed Cable: {round(jtheta2len(fixed_pos[i]))} mm")
        if num_fixed == 2: abs_err_axs[i].set_title(f"Length of Fixed Cables: {round(jtheta2len(fixed_pos[i]))} mm")
        # if i == 1:
            # abs_err_axs[i].legend(loc='lower left', bbox_to_anchor=(0.9, 0.8))
        abs_err_axs[i].legend()
        
        ax[i,0].set_ylim([-10, 10])
        ax[i,1].set_ylim([-7.5, 7.5])
        ax[i,2].set_ylim([-10, 10])
        for j in range(3):
            abs_err_curv0 = (pos_base_rot0[:,j] - est_pos_base_rot0[:,j])
            # ax[i,j].set_xticks(plot_len0[::2], labels=[f"{num:.2f}" for num in traj_[mask_curv_test[i*3],servos[0]][::2]])
            ax[i,j].set_xticks(np.linspace(-1, 1, tick_num), labels=[f"{num:.1f}" for num in np.concatenate((np.linspace(max_len, min_len, tick_half), np.linspace(min_len, max_len, tick_half)[1:]))])
            # ax[i,j].scatter(0, abs_err_curv0[0], label="Start", color="g")
            # ax[i,j].scatter(len(pos_base_rot0)-1, abs_err_curv0[-1], label="Finish", color="r")
            
            if not not_first:
                ax[i,j].scatter(plot_len0[0], abs_err_curv0[0], label="Start", color="g")
                ax[i,j].scatter(plot_len0[-1], abs_err_curv0[-1], label="Finish", color="r")
            ax[i,j].plot(plot_len0, abs_err_curv0, marker="*", label="Along Servo 1, " + plt_label)        

            abs_err_curv1 = (pos_base_rot1[:,j] - est_pos_base_rot1[:,j]) 
            ax[i,j].plot(plot_len1, abs_err_curv1, marker="*", label="Along Servo 2, " + plt_label)
            # ax[i,j].set_xticks(plot_len1[::2], labels=[f"{num:.2f}" for num in traj_[mask_curv_test[i*3],servos[0]][::2]])
            # ax[i,j].scatter(0, abs_err_curv1[0], color="g") # label="Start",
            ax[i,j].scatter(plot_len1[0], abs_err_curv1[0], color="g") # label="Start",
            ax[i,j].scatter(plot_len1[-1], abs_err_curv1[-1], color="r") # label="Finish",

            abs_err_curv2 = (pos_base_rot2[:,j] - est_pos_base_rot2[:,j]) 
            ax[i,j].plot(plot_len2, abs_err_curv2, marker="*", label="Along Servo 3, " + plt_label)
            # ax[i,j].set_xticks(plot_len2[::2], labels=[f"{num:.2f}" for num in traj_[mask_curv_test[i*3],servos[0]][::2]])
            ax[i,j].scatter(plot_len2[0], abs_err_curv2[0], color="g") # label="Start",
            # ax[i,j].scatter(0, abs_err_curv2[0], color="g") # label="Start",
            ax[i,j].scatter(plot_len2[-1], abs_err_curv2[-1], color="r") # label="Finish",
            # ax[i,j].scatter(len(pos_base_rot2)-1, abs_err_curv2[-1], color="r") # label="Finish",

            if vert_lines[1] < 1:
                # print(vert_lines)
                # ax[i,j].vlines(vert_lines, [ax[i,j].get_ylim()[0]]*2, [ax[i,j].get_ylim()[1]]*2, color='r', label='Spring Resting Length Position', linestyle='dotted')
                ax[i,j].vlines(fixed_pos_len, [ax[i,j].get_ylim()[0]]*2, [ax[i,j].get_ylim()[1]]*2, color='k', label='Fixed FR Length', linestyle='dotted')
            ax[i,j].set_ylabel(axis_labels[j])
            if i == 2: ax[i, j].set_xlabel('Moving Cable Length [mm]')
            if num_fixed == 1: ax[i,j].set_title(f"Length of Fixed Cable: {round(jtheta2len(fixed_pos[i]))} mm")
            if num_fixed == 2: ax[i,j].set_title(f"Length of Fixed Cables: {round(jtheta2len(fixed_pos[i]))} mm")
            # ax[i,j].hlines(0, 0, len(pos_base_rot0), linestyle='--', color='k')
            ax[i,j].hlines(0, -1, 1, linestyle='--', color='k')

    # ax[0,0].legend(loc='lower right', bbox_to_anchor=(0, 1))
    ax[0,2].legend(loc='lower left', bbox_to_anchor=(1.05, 0.7))

def plot_curv_test_2d_traj2(fixed_pos, num_fixed, traj, traj_, pos_base, est_pos_base, est_flag, ax, plt_label='test'):
    mask_curv_test = [[False] * len(traj)] * 9
        
    for i, fix_pos in enumerate(fixed_pos):
        for j in range(3):
            idx = [0,1,2]
            idx.remove(j)
            if num_fixed == 1:
                mask_curv_test[i*3+j] = ((traj[:,j] == fix_pos) & (traj[:,idx[0]] == traj[:,idx[1]])) # 1 cable fixed, 2 moving
            elif num_fixed == 2:
                mask_curv_test[i*3+j] = (traj[:,idx[0]] == fix_pos) & (traj[:,idx[1]] == fix_pos) # 2 cables fixed, 1 moving

    max_len = np.max(traj_)
    min_len = np.min(traj_)

    if num_fixed == 1: servos = [1, 2, 0]
    if num_fixed == 2: servos = [0, 1, 2]

    if num_fixed == 1:
        ang0 = 0
        ang1 = np.radians(-120)
        ang2 = np.radians(-240)
    else:
        ang0 = 0
        ang1 = np.radians(-120)
        ang2 = np.radians(-240)

    axis_labels = ['Position in x [mm]', 'Position in y [mm]', 'Position in z [mm]']
    for i in range(3):
        mask0 = mask_curv_test[i*3]
        pos_base_rot0 = np.array([np.array([[np.cos(ang0), -np.sin(ang0), 0],
                        [np.sin(ang0), np.cos(ang0), 0],
                        [0, 0, 1]]) @ p for p in pos_base[mask_curv_test[i*3]]])
        est_pos_base_rot0 = np.array([np.array([[np.cos(ang0), -np.sin(ang0), 0],
                        [np.sin(ang0), np.cos(ang0), 0],
                        [0, 0, 1]]) @ p for p in est_pos_base[mask_curv_test[i*3]]])
        plot_len0 = range(len(pos_base_rot0))
        neg_sec = traj_[mask0, servos[0]][np.insert(np.where(np.diff(traj_[mask0, servos[0]]) < 0)[0]+1, 0, 0)]
        pos_sec = traj_[mask0, servos[0]][np.where(np.diff(traj_[mask0, servos[0]]) >= 0)[0]+1]
        plot_len0 = np.concatenate((np.interp(neg_sec, [min_len, max_len], [0, -1]), np.interp(pos_sec, [min_len, max_len], [0, 1])))


        mask1 = mask_curv_test[i*3+1]
        pos_base_rot1 = np.array([np.array([[np.cos(ang1), -np.sin(ang1), 0],
                        [np.sin(ang1), np.cos(ang1), 0],
                        [0, 0, 1]]) @ p for p in pos_base[mask_curv_test[i*3+1]]])
        # est_pos_base_rot1 = np.array([np.array([[np.cos(ang1), -np.sin(ang1), 0],
        #                 [np.sin(ang1), np.cos(ang1), 0],
        #                 [0, 0, 1]]) @ p for p in est_pos_base[mask_curv_test[i*3+1]]])
        plot_len1 = range(len(pos_base_rot1))
        neg_sec = traj_[mask1, servos[1]][np.insert(np.where(np.diff(traj_[mask1, servos[1]]) < 0)[0]+1, 0, 0)]
        pos_sec = traj_[mask1, servos[1]][np.where(np.diff(traj_[mask1, servos[1]]) >= 0)[0]+1]
        plot_len1 = np.concatenate((np.interp(neg_sec, [min_len, max_len], [0, -1]), np.interp(pos_sec, [min_len, max_len], [0, 1])))

        mask2 = mask_curv_test[i*3+2]
        pos_base_rot2 = np.array([np.array([[np.cos(ang2), -np.sin(ang2), 0],
                        [np.sin(ang2), np.cos(ang2), 0],
                        [0, 0, 1]]) @ p for p in pos_base[mask_curv_test[i*3+2]]])
        # est_pos_base_rot2 = np.array([np.array([[np.cos(ang2), -np.sin(ang2), 0],
        #                 [np.sin(ang2), np.cos(ang2), 0],
        #                 [0, 0, 1]]) @ p for p in est_pos_base[mask_curv_test[i*3+2]]])
        plot_len2 = range(len(pos_base_rot2))
        neg_sec = traj_[mask2, servos[2]][np.insert(np.where(np.diff(traj_[mask2, servos[2]]) < 0)[0]+1, 0, 0)]
        pos_sec = traj_[mask2, servos[2]][np.where(np.diff(traj_[mask2, servos[2]]) >= 0)[0]+1]
        plot_len2 = np.concatenate((np.interp(neg_sec, [min_len, max_len], [0, -1]), np.interp(pos_sec, [min_len, max_len], [0, 1])))


        # if i == 0:
        #     test_fig, test_ax = plt.subplots(1, 2, figsize=(10,4))
        #     test_ax[0].plot(traj_[mask_curv_test[i]], label=['Cable 1', 'Cable 2', 'Cable 3'], marker='o', markersize=5)
        #     test_ax[0].legend()
        #     test_ax[1].plot(est_pos_base[mask_curv_test[i], 0], est_pos_base[mask_curv_test[i], 2]) #  marker='o', markersize=5
        #     s = test_ax[1].plot(pos_base[mask_curv_test[i], 0], pos_base[mask_curv_test[i], 2], label="Real Position") #  marker='o', markersize=5
        #     # plt.colorbar(s, ax=test_ax[1])

        for j in [0,1]:
            tick_num = 7 # odd
            tick_half = int(np.ceil(tick_num/2))

            if j == 0: axis = 0
            elif j == 1: axis = 2

            if est_flag:
                ax[i,j].plot(plot_len0, est_pos_base_rot0[:,axis], marker="o", linestyle="--", label="est")
            # abs_err_curv0 = (pos_base_rot0[:,j] - est_pos_base_rot0[:,j])
            # ax[i,j].set_xticks(plot_len0[::2], labels=[f"{num:.2f}" for num in traj_[mask_curv_test[i*3],servos[0]][::2]])
            ax[i,j].set_xticks(np.linspace(-1, 1, tick_num), labels=[f"{num:.2f}" for num in np.concatenate((np.linspace(max_len, min_len, tick_half), np.linspace(min_len, max_len, tick_half)[1:]))])
            # ax[i,j].scatter(0, pos_base_rot0[0,j], label="Start", color="g")
            # ax[i,j].scatter(len(pos_base_rot0)-1, pos_base_rot0[-1,j], label="Finish", color="r")
            ax[i,j].scatter(plot_len0[0], pos_base_rot0[0,axis], label="Start", color="g")
            ax[i,j].scatter(plot_len0[-1], pos_base_rot0[-1,axis], label="Finish", color="r")
            ax[i,j].plot(plot_len0, pos_base_rot0[:,axis], marker="*", label="Along Servo 1, " + plt_label)        


            # abs_err_curv1 = (pos_base_rot1[:,j] - est_pos_base_rot1[:,j])
            # ax[i,j].scatter(0, pos_base_rot1[0,j], color="g") # label="Start",
            # ax[i,j].scatter(len(pos_base_rot1)-1, pos_base_rot1[-1,j], color="r") # label="Finish",
            ax[i,j].plot(plot_len1, pos_base_rot1[:,axis], marker="*", label="Along Servo 2, " + plt_label)
            ax[i,j].scatter(plot_len1[0], pos_base_rot1[0,axis], label="Start", color="g")
            ax[i,j].scatter(plot_len1[-1], pos_base_rot1[-1,axis], label="Finish", color="r")

            # abs_err_curv2 = (pos_base_rot2[:,j] - est_pos_base_rot2[:,j])
            # ax[i,j].scatter(0, pos_base_rot2[0,j], color="g") # label="Start",
            # ax[i,j].scatter(len(pos_base_rot2)-1, pos_base_rot2[-1,j], color="r") # label="Finish",
            ax[i,j].plot(plot_len2, pos_base_rot2[:,axis], marker="*", label="Along Servo 3, " + plt_label)
            ax[i,j].scatter(plot_len2[0], pos_base_rot2[0,axis], label="Start", color="g")
            ax[i,j].scatter(plot_len2[-1], pos_base_rot2[-1,axis], label="Finish", color="r")

            ax[i,j].set_ylabel(axis_labels[axis])
            if i == 2 and num_fixed == 1: ax[i,j].set_xlabel('Length of Moving Cables [mm]')
            if i == 2 and num_fixed == 2: ax[i,j].set_xlabel('Length of Moving Cable [mm]')
            if num_fixed == 1: ax[i,j].set_title(f"Length of Fixed Cable: {jtheta2len(fixed_pos[i]):.2f} mm")
            if num_fixed == 2: ax[i,j].set_title(f"Length of Fixed Cables: {jtheta2len(fixed_pos[i]):.2f} mm")
            # ax[i,j].hlines(0, 0, len(pos_base_rot0), linestyle='--', color='k')
    ax[0,1].legend(loc='upper right', bbox_to_anchor=(1.25, 1.25))

from scipy.interpolate import make_interp_spline
def plot_vert_test(radius, pos_base, est_pos_base):
    est_pos_base = est_pos_base[pos_base[:,2] > 84]
    pos_base = pos_base[pos_base[:,2] > 84]
    if radius <= 0:
        pos_base_mask = [True] * len(pos_base)
    elif radius > 0: pos_base_mask = (np.linalg.norm(pos_base[:,:2], axis=1) < radius)

    # center
    mean_pos = np.mean(pos_base, axis=0)
    # print("mean pos:", mean_pos)
    pos_base[:, :2] -= mean_pos[:2]
    est_pos_base[:, :2] -= mean_pos[:2]

    pos_base_vert = pos_base[pos_base_mask]
    print(pos_base_vert.shape)
    est_pos_base_vert = est_pos_base[pos_base_mask]

    z_diff = np.mean(pos_base_vert[:,2] - est_pos_base_vert[:,2])
    pos_base_vert[:, 2] -= z_diff

    plt.figure(figsize=(10,4))
    plt.plot(pos_base_vert[:,2], label='Measured Position', marker='o', markersize=4)
    plt.plot(est_pos_base_vert[:,2], '--', label='CC Estimate')
    # plt.plot(est_pos_base_vert[:,2], '--', marker='o', markersize=4)
    plt.xlabel('Sample Points')
    plt.ylabel('z [mm]')
    plt.title('End Effector Height during test')
    plt.legend(loc='lower left', bbox_to_anchor=(.8, 1.01))

    pos_base_vert[:,2] += z_diff
    print("\nMean Absolute Error relative to Z in near vertical position: ", np.mean(np.linalg.norm(pos_base_vert - est_pos_base_vert, axis=1)))
    print("Standard Deviation relative to Z in near vertical position: ", np.std(np.linalg.norm(pos_base_vert - est_pos_base_vert, axis=1)))
    # plt.scatter(pos_base_vert[:,2], np.linalg.norm(pos_base_vert - est_pos_base_vert, axis=1), c=np.linalg.norm(pos_base_vert[:,:2], axis=1))
    # plt.title('Absolute Error relative to Z in near vertical position (xy norm colormapped)')

    prev_dist = [pos_base_vert[i,2] - pos_base_vert[i-1,2] for i in range(1, len(pos_base_vert))]
    # prev_dist = [copysign(1, pos_base_vert[i,2] - pos_base_vert[i-1,2]) for i in range(1, len(pos_base_vert))]
    # plt.figure()
    prev_dist = np.insert(prev_dist, 0, 0)
    # plt.scatter(pos_base_vert[:,2], np.linalg.norm(pos_base_vert - est_pos_base_vert, axis=1), c=prev_dist, s=15, alpha=.6)
    # plt.scatter(pos_base_vert[:,2], pos_base_vert[:,2] - est_pos_base_vert[:,2], s=15, alpha=.5) # c=prev_dist,
    # print(pos_base_vert.shape, z_error_centered.shape, prev_dist.shape)
    dir_mask = np.array([copysign(1, d) > 0 for d in prev_dist])
    z_set = np.array(sorted(set(est_pos_base[:,2])))
    # print(z_set)
    z_diff = z_set[1] - z_set[0]
    z_bins = [z_set[0] - z_diff/2] + list(z_set + z_diff/2)
    # print(z_bins)

    bin_means_z = [[], []]
    bin_means_err = [[], []]

    z_error_centered = [
        pos_base[dir_mask,2] - est_pos_base[dir_mask,2] - np.mean(pos_base_vert[dir_mask,2] - est_pos_base_vert[dir_mask,2]),
        pos_base[~dir_mask,2] - est_pos_base[~dir_mask,2] - np.mean(pos_base_vert[~dir_mask,2] - est_pos_base_vert[~dir_mask,2])
    ]

    for i in range(len(z_bins)-1):
        mask = (
            (pos_base_vert[dir_mask,2] >= z_bins[i]) & (pos_base_vert[dir_mask,2] < z_bins[i+1]),
            (pos_base_vert[~dir_mask,2] >= z_bins[i]) & (pos_base_vert[~dir_mask,2] < z_bins[i+1])
        )
        bin_means_z[0].append(np.mean(pos_base_vert[dir_mask][mask[0],2]))
        bin_means_z[1].append(np.mean(pos_base_vert[~dir_mask][mask[1],2]))
        bin_means_err[0].append(np.mean(z_error_centered[0][mask[0]]))
        bin_means_err[1].append(np.mean(z_error_centered[1][mask[1]]))

    # include top for going down spline
    bin_means_z[1].insert(0, bin_means_z[0][-1])
    bin_means_err[1].insert(0, bin_means_err[0][-1])

    bin_means_err = [np.array(arr) for arr in bin_means_err]
    bin_means_z = [np.array(arr) for arr in bin_means_z]
    clean_mask = [~np.isnan(bin_means_z[i]) & ~np.isnan(bin_means_err[i]) for i in range(2)]
    bin_means_z = [bin_means_z[i][clean_mask[i]] for i in range(2)]
    sort_idx = [arr.argsort() for arr in bin_means_z]
    bin_means_z = [bin_means_z[i][sort_idx[i]] for i in range(2)]
    bin_means_err = [bin_means_err[i][clean_mask[i]][sort_idx[i]] for i in range(2)]

    plt.figure()

    xnew_up = np.linspace(bin_means_z[0].min(), bin_means_z[0].max(), 100)
    spl_up = make_interp_spline(bin_means_z[0], bin_means_err[0], k=3)  # Cubic spline
    y_smooth_up = spl_up(xnew_up)

    xnew_down = np.linspace(bin_means_z[1].min(), bin_means_z[1].max(), 100)
    spl_down = make_interp_spline(bin_means_z[1], bin_means_err[1], k=3)  # Cubic spline
    y_smooth_down = spl_down(xnew_down)

    plt.vlines(125.5, -1, 1, label='Robot\'s Resting Length', alpha=0.75, linestyle='dotted', linewidth=2)
    plt.scatter(pos_base_vert[dir_mask, 2], z_error_centered[0], color='g', label='Moving Up', alpha=0.5) # c=prev_dist
    plt.scatter(pos_base_vert[~dir_mask, 2], z_error_centered[1], color='r', label='Moving Down', alpha=0.5) # c=prev_dist
    plt.plot(xnew_up, y_smooth_up, label='Fitted Line, Moving Up', color='g', linestyle='--')
    plt.plot(xnew_down, y_smooth_down, label='Fitted Line, Moving Down', color='r', linestyle='--')
    # plt.plot(bin_means_z[0], bin_means_err[0], label='Fitted Line, Moving Up', color='b', linestyle='--')
    # plt.plot(bin_means_z[1], bin_means_err[1], label='Fitted Line, Moving Down', color='orange', linestyle='--')

    # ax1[2].hlines(np.mean(pos_base[:,2] - est_pos_base[:,2]), *plt.xlim(), linestyles='--')
    # plt.title('Colormap: Displacement from prev point in z-axis')
    plt.title('Constant Curvature Model Error during test')
    plt.ylabel('Absolute Error in z-axis [mm]')
    plt.xlabel('End Effector Height [mm]')
    plt.ylim([-1,1])
    # plt.xlim([z_set[0]-2, z_set[-1]+2])
    plt.grid(visible=True)
    # ax1[2].colorbar()
    # plt.colorbar()
    plt.legend(loc='upper left')

    fig1, ax1 = plt.subplots(1, 3, figsize=(15,5))
    fig1.suptitle("Absolute Error for each coordinate relative to z position")
    prev_dist = [pos_base_vert[i,2] - pos_base_vert[i-1,2] for i in range(1, len(pos_base_vert))]
    # prev_dist = [copysign(1, pos_base_vert[i,2] - pos_base_vert[i-1,2]) for i in range(1, len(pos_base_vert))]
    # plt.figure()
    prev_dist.insert(0, 0)
    # plt.scatter(pos_base_vert[:,2], np.linalg.norm(pos_base_vert - est_pos_base_vert, axis=1), c=prev_dist, s=15, alpha=.6)
    mapp1 = ax1[2].scatter(pos_base_vert[:,2], pos_base_vert[:,2] - est_pos_base_vert[:,2], c=prev_dist, s=15, alpha=.5)
    # ax1[2].hlines(np.mean(pos_base[:,2] - est_pos_base[:,2]), *plt.xlim(), linestyles='--')
    ax1[2].set_title('Colormap: Displacement from prev point in x-axis')
    ax1[2].set_ylabel('Absolute Error in z-axis [mm]')
    ax1[2].set_xlabel('z [mm]')
    ax1[2].grid(visible=True)
    # ax1[2].colorbar()

    # plt.figure()
    # prev_dist = [pos_base_vert[i,2] - pos_base_vert[i-1,2] for i in range(1, len(pos_base_vert))]
    # prev_dist.insert(0, 0)
    # plt.scatter(pos_base_vert[:,2], np.linalg.norm(pos_base_vert - est_pos_base_vert, axis=1), c=prev_dist, s=15, alpha=.6)
    mapp2 = ax1[0].scatter(pos_base_vert[:,2], pos_base_vert[:,0] - est_pos_base_vert[:,0], c=prev_dist, s=15, alpha=.6)
    ax1[0].set_title('Colormap: Displacement from prev point in y-axis')
    ax1[0].set_ylabel('Absolute Error in x-axis')
    ax1[0].set_xlabel('z [mm]')
    ax1[0].grid(visible=True)
    # ax1[0].colorbar()
    fig1.colorbar(mapp2, ax=ax1[0])
    
    # plt.figure()
    # prev_dist = [pos_base_vert[i,2] - pos_base_vert[i-1,2] for i in range(1, len(pos_base_vert))]
    # prev_dist.insert(0, 0)
    # plt.scatter(pos_base_vert[:,2], np.linalg.norm(pos_base_vert - est_pos_base_vert, axis=1), c=prev_dist, s=15, alpha=.6)
    mapp3 = ax1[1].scatter(pos_base_vert[:,2], pos_base_vert[:,1] - est_pos_base_vert[:,1], c=prev_dist, s=15, alpha=.6)
    ax1[1].set_title('Colormap: Displacement from prev point in z-axis')
    ax1[1].set_ylabel('Absolute Error in y-axis')
    ax1[1].set_xlabel('z [mm]')
    ax1[1].grid(visible=True)
    # ax1[1].colorbar()
    fig1.colorbar(mapp3, ax=ax1[1])
    
    fig = plt.figure()
    ax = plt.subplot()
    # grid_circ = [plt.Circle((0,0), i / 5 * np.max([pos_base_vert[:,0], pos_base_vert[:,1]]),  linewidth=0.5, fill=False) for i in range(10)]
    grid_circ = [plt.Circle((0,0), i,  linewidth=0.5, fill=False) for i in range(10)]
    plt.scatter(pos_base_vert[:,0], pos_base_vert[:,1], c = pos_base_vert[:,2])
    # plt.scatter(pos_base_vert[:,0], pos_base_vert[:,1], c = np.linalg.norm(pos_base_vert - est_pos_base_vert, axis=1))
    plt.colorbar()
    lims = max(max(np.abs(pos_base_vert[:,0])), max(np.abs(pos_base_vert[:,1]))) * 1.2
    plt.xlim(-lims, lims)
    plt.ylim(-lims, lims)
    for circ in grid_circ:
        ax.add_patch(circ)
    plt.grid(visible=True)
    plt.title('xy position with z [mm] colormapped')
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    # plt.title('xy position with position error colormapped')
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x, y, z = zip(*pos_base_vert)
    # ax.scatter(x, y, z, c=np.linalg.norm(pos_base_vert - est_pos_base_vert, axis=1), marker='o', label='')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # # ax.plot([0,0], [0,0], [0,200], linestyle='dashed')
    # # ax.plot([0,0], [0,0], [np.min(est_pos_base[:, 2] - 36),k])
    # # draw_robot(ax, alpha_mult=0.2, draw_cables=True, travel=True)
    # plt.title('xyz position with position error colormapped')

def plot_rel_err(pos_base, est_pos_base, rel_err_norm, mean_traj_, xyz_flag=False, d3d_flag=False):
    plot_len = range(len(pos_base))
    plt.figure()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # mean_traj_dev = abs(mean_traj_ - np.mean(mean_traj_))
    # mean_traj_dev = mean_traj_ - np.mean(mean_traj_)
    mean_traj_dev = mean_traj_ - 125.5
    # prev_dist = [1 if pos_base[i,2] > pos_base[i-1,2] else 0 for i in range(1, len(pos_base))]
    # prev_dist = [np.linalg.norm(pos_base[i,:] - pos_base[i-1,:]) for i in range(1, len(pos_base))]
    # prev_dist.insert(0, 0)
    # plt.scatter(plot_len, rel_err_norm*100, c=prev_dist)
    # plt.scatter(np.linalg.norm(pos_base[:,:2], axis=1), rel_err_norm*100, c=mean_traj_)
    # print(mean_traj_.shape, rel_err_norm.shape, np.linalg.norm(pos_base[:,:2], axis=1).shape)
    plt.scatter(np.linalg.norm(pos_base[:,:2], axis=1), rel_err_norm*100, alpha=0.8, c=mean_traj_dev)
    # plt.scatter(prev_dist, rel_err_norm*100)
    plt.colorbar()
    # plt.title("Relative Error w/ Distance to Previous WayPoint colormapped")
    # plt.title("Relative Error relative to Distance to Previous WayPoint")
    # plt.grid(visible=True)
    # plt.hlines(np.mean(rel_err_norm)*100, min(mean_traj_)*1.05, max(mean_traj_)*1.05, label='Mean', linestyles='--', colors='r')
    plt.xlabel("Distance to z-axis [mm]")
    plt.ylabel("Resting Length Relative Error [%]")
    # plt.title("Absolute Difference to Resting Length of Avg Cable Length [mm] (colormapped)", wrap=True)
    plt.title("Difference of Mean Cable Length to Resting Length [mm] (colormapped)", wrap=True)
    plt.grid()
    # plt.hlines(np.mean(rel_err_norm)*100, 0, len(traj), label='Mean', linestyles='--', colors='r')
    # plt.hlines(np.mean(rel_err_norm)*100, 0, max(prev_dist), label='Mean', linestyles='--', colors='r')
    # plt.ylim(-.2, np.mean(rel_err_norm) * 500)
    # plt.legend()

    # fig = plt.figure()
    # ax = plt.subplot()
    # grid_circ = [plt.Circle((0,0), i / 5 * np.max([pos_base[:,0], pos_base[:,1]]),  linewidth=0.5, fill=False) for i in range(10)]
    # plt.scatter(pos_base[:,0], pos_base[:,1], c = pos_base[:,2])
    # # plt.scatter(pos_base[:,0], pos_base[:,1], c = np.linalg.norm(pos_base - est_pos_base, axis=1))
    # plt.colorbar()
    # lims = max(max(np.abs(pos_base[:,0])), max(np.abs(pos_base[:,1]))) * 1.2
    # plt.xlim(-lims, lims)
    # plt.ylim(-lims, lims)
    # for circ in grid_circ:
    #     ax.add_patch(circ)
    # plt.grid(visible=True)
    # plt.title('xy position with z (mm) colormapped')
    # plt.xlabel("x (mm)")
    # plt.ylabel("y (mm)")

    if xyz_flag:
        plt.figure()
        plt.scatter(plot_len, pos_base[:,0], label='Polaris', marker='o')
        plt.scatter(plot_len, est_pos_base[:,0], label='PCC Estimation', marker='o')
        plt.grid(visible=True)
        # plt.plot(pos_base[:,0] - est_pos_base[:, 0])
        plt.legend()
        plt.title('x')
        plt.figure()
        plt.scatter(plot_len, pos_base[:,1], label='Polaris', marker='o')
        plt.scatter(plot_len, est_pos_base[:,1], label='PCC Estimation', marker='o')
        plt.grid(visible=True)
        # plt.plot(pos_base[:,1] - est_pos_base[:, 1])
        plt.legend()
        plt.title('y')
        plt.figure()
        plt.scatter(plot_len, pos_base[:,2], label='Polaris', marker='o')
        plt.scatter(plot_len, est_pos_base[:,2], label='PCC Estimation', marker='o')
        plt.grid(visible=True)
        # plt.plot(pos_base[:,2] - est_pos_base[:, 2])
        plt.title('z')
        plt.legend()
    
    if d3d_flag:
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        half = len(pos_base) // 3
        # half = 100
        x, y, z = zip(*pos_base[:half])
        ax.scatter(x, y, z, c='b', marker='o', label='Polaris', alpha=0.5)
        x, y, z = zip(*est_pos_base[:half])
        ax.scatter(x, y, z, c='r', marker='o', label='PCC Estimation', alpha=0.5)
        ax.set_ylabel('y [mm]')
        ax.set_xlabel('x [mm]')
        ax.set_zlabel('z [mm]')
        plt.legend()

def plot_abs_err(pos_base, est_pos_base, abs_err_norm, mean_traj_, xyz_flag=False, annotate_flag=False, d3d_flag=False):
    plot_len = range(len(pos_base))
    plt.figure()
    # prev_dist = [1 if pos_base[i,2] > pos_base[i-1,2] else 0 for i in range(1, len(pos_base))]
    # prev_dist = [np.linalg.norm(pos_base[i,:] - pos_base[i-1,:]) for i in range(1, len(pos_base))]
    # prev_dist.insert(0, 0)
    # plt.scatter(plot_len, rel_err_norm*100, c=prev_dist)
    plt.plot([], [], label=f'Mean Error = {np.mean(abs_err_norm):.3f}mm', linestyle='', color='w', alpha=0)
    # plt.scatter(np.linalg.norm(pos_base[:,:2], axis=1), abs_err_norm, c=mean_traj_)
    # plt.scatter(mean_traj_, abs_err_norm, c=np.linalg.norm(pos_base[:,:2], axis=1), alpha=0.7)
    print(pos_base.shape, abs_err_norm.shape, mean_traj_.shape)
    plt.scatter(np.linalg.norm(pos_base[:,:2], axis=1), abs_err_norm, c=mean_traj_, alpha=0.7)
    # plt.scatter(prev_dist, rel_err_norm)
    plt.colorbar()
    # plt.title("Relative Error w/ Distance to Previous WayPoint colormapped")
    # plt.title("Relative Error relative to Distance to Previous WayPoint")
    # plt.title(r"Colormap: average cable length [mm]")
    plt.grid(visible=True)
    # plt.hlines(np.mean(abs_err_norm), 85, 145, label='Mean', linestyles='--', colors='r')
    if annotate_flag:
        ax = plt.gca()
        for i in plot_len:
            ax.annotate(i+1, (np.linalg.norm(pos_base[:,:2], axis=1)[i], abs_err_norm[i]))

    plt.xlabel(r"Distance to $z$-axis [mm]")
    plt.title("Mean Flexible Rod length [mm] (colormapped)")
    # plt.title(r"Colormap: distance to $z$-axis [mm]")
    # plt.xlabel(r"Mean cable length [mm]")
    plt.ylabel(r"CC Absolute Error [mm]")

    y_lims = plt.gca().get_ylim()
    # plt.vlines(125.5, y_lims[0], y_lims[1], linestyle='dotted', color='k', label='Robot Resting Length = 125.5 mm')
    # plt.hlines(np.mean(abs_err_norm), 0, len(traj), label='Mean', linestyles='--', colors='r')
    # plt.hlines(np.mean(abs_err_norm), 0, max(prev_dist), label='Mean', linestyles='--', colors='r')
    # plt.ylim(-.2, np.mean(abs_err_norm) * 500)
    plt.legend()

    plt.figure(figsize=(11,9)).add_subplot(121)
    plt.gcf().tight_layout(pad=5, h_pad=5)
    plt.suptitle("CC Absolute Error (colormapped) [mm]")
    plt.grid()
    plt.scatter(pos_base[:,0], pos_base[:,1], c=abs_err_norm, alpha=0.6, s=15)
    plt.gca().set_aspect('equal')
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.colorbar()
    plt.title("Top Down View ($xy$ plane)")
    plt.gcf().add_subplot(222)
    plt.grid()
    plt.scatter(pos_base[:,0], pos_base[:,2], c=abs_err_norm, alpha=0.6, s=15)
    plt.gca().set_aspect('equal')
    plt.xlabel('x [mm]')
    plt.ylabel('z [mm]')
    plt.title("Side View ($xz$ plane)")
    plt.gcf().add_subplot(224)
    plt.grid()
    plt.scatter(pos_base[:,1], pos_base[:,2], c=abs_err_norm, alpha=0.6, s=15)
    plt.gca().set_aspect('equal')
    plt.xlabel('y [mm]')
    plt.ylabel('z [mm]')
    plt.title("Side View ($yz$ plane)")
    if xyz_flag:
        plt.figure()
        plt.scatter(plot_len, pos_base[:,0], label='Polaris', marker='o')
        plt.scatter(plot_len, est_pos_base[:,0], label='PCC Estimation', marker='o')
        plt.grid(visible=True)
        # plt.plot(pos_base[:,0] - est_pos_base[:, 0])
        plt.legend()
        plt.title('x')
        plt.figure()
        plt.scatter(plot_len, pos_base[:,1], label='Polaris', marker='o')
        plt.scatter(plot_len, est_pos_base[:,1], label='PCC Estimation', marker='o')
        plt.grid(visible=True)
        # plt.plot(pos_base[:,1] - est_pos_base[:, 1])
        plt.legend()
        plt.title('y')
        plt.figure()
        plt.scatter(plot_len, pos_base[:,2], label='Polaris', marker='o')
        plt.scatter(plot_len, est_pos_base[:,2], label='PCC Estimation', marker='o')
        plt.grid(visible=True)
        # plt.plot(pos_base[:,2] - est_pos_base[:, 2])
        plt.title('z')
        plt.legend()
    
    if d3d_flag:
        # Create a 3D plot
        fig = plt.figure()
        fig.suptitle("CC Absolute Error [mm] (colormapped)")
        # x, y, z = zip(*est_pos_base)
        # ax.scatter(x, y, z, c='r', marker='o', label='PCC Estimation')
        # plt.legend()

        for i in range(2,3):
            ax = fig.add_subplot(111, projection='3d')
            x, y, z = zip(*pos_base)
            # ax.scatter(x, y, z, c='b', marker='o', label='Polaris')
            sc = ax.scatter(x, y, z, c=np.linalg.norm(pos_base - est_pos_base, axis=1), marker='o', label='Polaris', s=10)
            if i == 2: plt.colorbar(sc)
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('y [mm]')
            ax.set_zlabel('z [mm]')
            # ax.view_init(azim=120*i)
            ax.set_title(f"View from Servo {i+1}")
        
        # fig.tight_layout(pad=1.6, h_pad=1.2, w_pad=1.7, rect=(0, 0.1, 1, 1))

def plot_3d(pos_base, est_pos_base, ax, idx, pcc_flag=False, annotate_flag=True):
    if pcc_flag:
        x, y, z = zip(*est_pos_base)
        ax.scatter(x, y, z, c='r', marker='*', label='PCC Estimation')
    x, y, z = zip(*pos_base)
    ax.scatter(x, y, z, marker='o', label=f'Polaris {idx}')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_zlabel('z [mm]')
    if annotate_flag and pcc_flag:
        for i in range(len(pos_base)):
            ax.text(x[i], y[i], z[i], str(i+1))
    # plt.legend()

def plot_grid(traj):
    plot_len = range(len(traj))
    fig1, ax1 = plt.subplots(3, 1, figsize=(8,6))
    for i in range(3):
        ax1[i].plot(plot_len, traj[:,i], marker='*')
