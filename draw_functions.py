import numpy as np
import sys, os

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
# from mpl_toolkits.mplot3d import Axes3D
from kinematics_functions import T_beModule, jtheta2len

def draw_cylinder(ax, x_center, y_center, z_center, radius, height, num_sections, alphas):
    theta = np.linspace(0, 2*np.pi, 100)
    z = np.linspace(0, height, num_sections) + z_center
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + x_center + np.linspace(0, 0.5, num_sections)[:, np.newaxis]
    y_grid = radius * np.sin(theta_grid) + y_center + np.linspace(0, 0.5, num_sections)[:, np.newaxis]

    for i in range(num_sections-1):
        z_values = np.stack((z_grid[i], z_grid[i+1]))
        x_values = np.stack((x_grid[i], x_grid[i+1]))
        y_values = np.stack((y_grid[i], y_grid[i+1]))
        ax.plot_surface(x_values, y_values, z_values, alpha=alphas[i], color='b')

def draw_vert_cylinder(ax, x_center, y_center, z_center, radius, height, alpha, color, outline=False, top=False):
    theta = np.linspace(0, 2*np.pi, 150)
    z = np.linspace(0, height, 2) + z_center
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + x_center
    y_grid = radius * np.sin(theta_grid) + y_center
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=alpha, color=color)
    if top:
        top_x = np.vstack((x_grid[0], np.zeros_like(x_grid[0])))
        top_y = np.vstack((y_grid[0], np.zeros_like(y_grid[0])))
        top_z = np.vstack((z_grid[1], z_grid[1]))
        # print(top_z.shape)
        ax.plot_surface(top_x, top_y, top_z, alpha=alpha, color=color)
    if outline:
        ax.plot_wireframe(x_grid, y_grid, z_grid, color='black', alpha=alpha, linewidth=0.5)

def draw_sphere(ax, x_center, y_center, z_center, radius, alpha, color, outline=False):
    u = np.linspace(0, 2 * np.pi, 150)
    v = np.linspace(0, np.pi, 150)
    x = x_center + radius * np.outer(np.cos(u), np.sin(v))
    y = y_center + radius * np.outer(np.sin(u), np.sin(v))
    z = z_center + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=alpha, color=color)
    if outline:
        ax.plot_wireframe(x, y, z, color='black', alpha=.8)

def draw_robot(ax, alpha_mult=0.8, draw_cables=False, travel=False):
    R = 21
    markerR = 4.4
    screwR = 1.5
    h1 = 4.2
    h2 = 4.2
    # h3 = 7.1 + 7.6
    h3 = 7 # + 2.9 # + 26
    h4 = 3
    h5 = 26 - markerR
    H = 40.6
    spring_color = 'turquoise'; spring_alpha = 0.4 * alpha_mult
    ring_color = '0.95'; ring_alpha = 0.8 * alpha_mult
    topring_color = '0.4'; topring_alpha = 0.6 * alpha_mult
    screw_color ='0.2'; screw_alpha = 0.8 * alpha_mult
    marker_color = '0.6'; marker_alpha = 0.8 * alpha_mult

    draw_vert_cylinder(ax, 0, 0, 0, R, h1, ring_alpha, ring_color, top=True)
    draw_vert_cylinder(ax, 0, 0, h1, R, H, spring_alpha, spring_color)
    draw_vert_cylinder(ax, 0, 0, h1 + H, R, h2, ring_alpha, ring_color, top=True)
    draw_vert_cylinder(ax, 0, 0, h1 + H + h2, R, H, spring_alpha, spring_color)
    draw_vert_cylinder(ax, 0, 0, h1 + H + h2 + H, R, h3, ring_alpha, ring_color, top=True)
    draw_vert_cylinder(ax, 0, 0, h1 + H + h2 + H + h3, R, h4, topring_alpha, topring_color, top=True)
    draw_vert_cylinder(ax, 0, 0, h1 + H + h2 + H + h3 + h4, screwR, h5, screw_alpha, screw_color)
    draw_sphere(ax, 0, 0, h1 + H + h2 + H + h3 + h4 + h5 + markerR, markerR, marker_alpha, marker_color)

    if draw_cables:
        cable_len = h1 + H + h2 + H + h3 / 2
        tip_rest = h1 + H + h2 + H + h3 + h4 + h5 + markerR
        cableR = 18.75
        cable_pos = np.array([[-cableR, 0, -5], [-cableR, 0, cable_len]])
        rot = np.linspace(0, 4*np.pi/3, 3)
        if travel:
            cable_pos1 = np.array([[-cableR, 0, 0], [-cableR, 0, 50]])
            cable_pos2 = np.array([[-cableR, 0, 50], [-cableR, 0, 108]])
            tip_max = tip_rest + (108 - cable_len)
            tip_min = tip_rest - (cable_len - 50)
        for i in range(3):
            rot = np.array([[np.cos(2*np.pi/3 * i), -np.sin(2*np.pi/3 * i), 0],
                            [np.sin(2*np.pi/3 * i), np.cos(2*np.pi/3 * i), 0],
                            [0, 0, 1]]).T
            if travel:
                pts = cable_pos @ rot
                pts1 = cable_pos1 @ rot
                pts2 = cable_pos2 @ rot
                ax.plot(*pts1.T, color='0.3', alpha=alpha_mult*1.1*0.7) #, marker='o'
                ax.plot(*pts2.T, color='0.3', alpha=alpha_mult*1.1, linestyle='dashed', label='Cable Travel') #, marker='o'
                ax.scatter(*pts[1].T, color='0.3', label='Resting Length')
                ax.scatter(*pts2[1].T, color='r', label='Maximum Length')
                ax.scatter(*pts1[1].T, color='g', label='Minimum Length')
                draw_sphere(ax, 0, 0, tip_min, markerR, alpha=0.8*marker_alpha, color='g')
                draw_sphere(ax, 0, 0, tip_max, markerR, alpha=0.8*marker_alpha, color='r')
                if i==0: ax.legend(loc='upper left', bbox_to_anchor=(-0.05, 1))
            else:
                pts = cable_pos @ rot
                ax.plot(*pts.T, color='0.3', alpha=0.6*alpha_mult*1.1) #, marker='o'
    # print( h1 + H + h2 + H + h3 + h4 + h5 + markerR)
    ax.set_aspect('equal')


def draw_work_volume(ax, alpha=0.2):
    n = 20
    vals = np.linspace(750, 2250, n, dtype=int)
    x, y, z = np.meshgrid(vals, vals, vals, indexing='ij')
    mesh = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    # print(z.shape)
    tol = 50
    condition = (
        ((np.abs(mesh[:, 0] + mesh[:, 1] - 4200) < tol) | (np.abs(mesh[:, 2] - 1220) < tol)) &
        ((np.abs(mesh[:, 1] + mesh[:, 2] - 4200) < tol) | (np.abs(mesh[:, 0] - 1220) < tol)) &
        ((np.abs(mesh[:, 0] + mesh[:, 2] - 4200) < tol) | (np.abs(mesh[:, 1] - 1220) < tol)) |
        (mesh[:, 0] == 2250) | (mesh[:, 0] == 750)  |
        (mesh[:, 1] == 2250) | (mesh[:, 1] == 750)  |
        (mesh[:, 2] == 2250) | (mesh[:, 2] == 750)
    )

    mesh_trimmed = mesh[condition]
    lens = jtheta2len(mesh_trimmed)
    points = np.array([T_beModule(p, [], 0, 0)[:3, 3] for p in lens])
    ax.plot(*points.T, alpha=alpha, linewidth=0, marker='.')

def draw_work_volume_error(ax, alpha=0.2):
    n = 20
    vals = np.linspace(750, 2250, n, dtype=int)
    x, y, z = np.meshgrid(vals, vals, vals, indexing='ij')
    mesh = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    # print(z.shape)
    tol = 50
    condition = (
        ((np.abs(mesh[:, 0] + mesh[:, 1] - 4200) < tol) | (np.abs(mesh[:, 2] - 1220) < tol)) &
        ((np.abs(mesh[:, 1] + mesh[:, 2] - 4200) < tol) | (np.abs(mesh[:, 0] - 1220) < tol)) &
        ((np.abs(mesh[:, 0] + mesh[:, 2] - 4200) < tol) | (np.abs(mesh[:, 1] - 1220) < tol)) |
        (mesh[:, 0] >= 2250) | (mesh[:, 0] <= 750)  |
        (mesh[:, 1] >= 2250) | (mesh[:, 1] <= 750)  |
        (mesh[:, 2] >= 2250) | (mesh[:, 2] <= 750)
    )

    mesh_trimmed = mesh[~condition]
    lens = jtheta2len(mesh_trimmed)
    rand_err = np.random.rand(len(lens), 3)
    rand_err = (rand_err / np.linalg.norm(rand_err, axis=1)[:, np.newaxis])
    lens_w_error = lens.copy() + rand_err
    # lens_w_error = lens.copy() + [0, 0, 1]
    # lens_w_error = lens.copy() + [1, 1, 1]
    # lens_w_error = lens.copy() + [1, 1, 0]
    # print(len(lens))
    # print(lens[:5], '\n', lens_w_error[:5])
    points = np.array([T_beModule(p, [], 0, 0)[:3, 3] for p in lens])
    points_w_error = np.array([T_beModule(p, [], 0, 0)[:3, 3] for p in lens_w_error])
    ax.plot(*points.T, alpha=alpha, linewidth=0, marker='.')
    ax.plot(*points_w_error.T, alpha=alpha, linewidth=0, marker='.', color='red')
    abs_err = np.linalg.norm(points - points_w_error, axis=1)

    ret = np.array([np.mean(abs_err), np.max(abs_err), np.min(abs_err)])
    print('mean err', ret[0])
    print('max err', ret[1])
    print('min err', ret[2])
    return ret


if __name__ == '__main__':
    # Create a figure and a 3D axis
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot()

    # Parameters for the cylinder
    x_center = 0
    y_center = 0
    z_center = 0
    radius = 1
    height = 2
    num_sections = 5
    alphas = [0.2, 0.4, 0.6, 0.8, 1.0]  # Example alphas for each section

    # Draw the cylinder with different alphas for each section
    # draw_cylinder(ax, x_center, y_center, z_center, radius, height, num_sections, alphas)
    # draw_vert_cylinder(ax, x_center, y_center, z_center, radius, height, 1, 'b')
    # draw_robot(ax)
    # draw_work_volume(ax)
    # ax.set_title("Estimated Work Volume, using PCC")
    # ax.set_xlabel('x [mm]')
    # ax.set_ylabel('y [mm]')
    # ax.view_init(elev=15, azim=35)

    avg_err = np.empty((10, 3), dtype=float)
    for i in range(10):
        avg_err[i,:] = draw_work_volume_error(ax)

    print('avg_mean_err', np.mean(avg_err[:,0]))
    print('avg_max_err', np.mean(avg_err[:,1]))
    print('avg_min_err', np.mean(avg_err[:,2]))

    # draw_robot(ax, alpha_mult=0.6, draw_cables=True, travel=True)
    # ax.set_title("Estimated Vertical Travel")
    # ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # ax.set_xticks([])
    # ax.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
    # ax.set_yticks([])
    # ax.tick_params(axis='z', pad=8)
    # ax.view_init(elev=2, azim=-1)

    # ax.set_zlabel('z [mm]', labelpad=20.)
    # ax.set_zlim([0, 150])
    # ax.set_aspect('equal')

    # traj = np.array([2000, 1220, 1220])
    # print(T_beModule(jtheta2len(traj), [], 0, 0)[:3, 3])
    # ax.plot(*T_beModule(jtheta2len(traj), [], 0, 0)[:3, 3], 'k', marker='o')

    if tex_plots:
        folder = "./_plots"
        if not os.path.exists(folder): os.mkdir(folder)
        title = "est_work_vol"
        fig.savefig(os.path.join(folder, title) + ".pgf")
        # plt.show()
    # else:
    #     plt.show()
