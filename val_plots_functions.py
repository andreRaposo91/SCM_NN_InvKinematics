import os, sys
import tkinter
from tkinter import filedialog
import numpy as np
import seaborn as sns
from math import copysign, ceil

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
plt.rcParams.update({'font.size': 12})
from math import copysign

from data_functions import parse_cont_dataset, polaris2base
from kinematics_functions import T_beModule
from point_clouds import generate_square, generate_circle, generate_coil
from draw_functions import draw_robot

MODEL_ORDER = ('CC', 'FNN3', 'FNN6', 'RNN', 'FNN3-CC', 'FNN6-CC', 'RNN-CC')

def display_model_name(name):
    return name.split('_')[1].upper().replace('PCC', 'CC')

def parse_log(log_filepath):
    runs = {}
    with open(log_filepath, 'r') as file:
        for i, line in enumerate([l for l in file.readlines() if not l.startswith('#')]):
            empty_keys = list(runs.keys())
            key_values = line.strip().split(";")
            for kv in key_values:
                key, value = kv.strip().split(':', 1)
                key = key.strip()
                value = value.strip()
                if key not in runs:
                    runs[key] = [""] * i
                runs[key].append(value)
                if key in empty_keys:
                    empty_keys.remove(key)
            for k in empty_keys:
                runs[k].append("")

    return runs

import pandas as pd
def validation_analysis(run_log_path="", run_folder="", zlims=(0,0), save_plot=False):

    if run_folder == "":
        run_folder = "./val/"
    if run_log_path == "":
        run_log_path = os.path.join(run_folder, "run_log.txt")

    runs = parse_log(run_log_path)
    df_runs = pd.DataFrame(runs)
    df_runs["inv_kin_model"] = df_runs["inv_kin_model"].apply(display_model_name)
    print(df_runs.shape)
    # print(df_runs.head())
    # sys.exit()
    # print(runs.keys())
    figs = []
    tests = list(dict.fromkeys(runs["traj command"]))
    models = [model for model in MODEL_ORDER
              if model in df_runs["inv_kin_model"].unique()]
    pts = list(set(runs["pause_time"]))
    print(models)
    runs_files = [(run_folder + file, runs["timestamp"].index(file.split('_')[-1][:-4])) for file in os.listdir(run_folder) if file.split('_')[-1][:-4] in runs['timestamp']]
    runs_idxs = list(range(df_runs.shape[0]))
    [runs_idxs.remove(idx) for _, idx in runs_files]
    if runs_idxs:
        print("Missing files timestamps", df_runs.loc[runs_idxs, "timestamp"])
    # sys.exit()
    
    figs = [plt.figure(figsize=(6*len(tests),6))] + [plt.figure(figsize=(6*len(models),6)) for _ in range(len(tests)*2)]
    scatter_axs = [[fig.add_subplot(1, len(models), i+1, projection="3d") for i in range(len(models))] for fig in figs[1:len(tests)+1]]
    error_axs = [[fig.add_subplot(2, len(models), i+1) for i in range(len(models)*2)] for fig in figs[len(tests)+1:]]
    # print(error_axs[0])
    # print(len(error_axs[0]))
    bar_axs = [figs[0].add_subplot(1, len(tests), i+1) for i in range(len(tests))] # Add bar plot subplots
    y_lim = 15

    targets = []

    for i, t in enumerate(tests):
        targets.append(eval(t))
        print(f"Test {i+1}: {t}, len: {len(targets[i])} points")
        figs[i+1].suptitle(f"3D - Test {i+1}: {t}")
        figs[len(tests)+i+1].suptitle(f"Error - Test {i+1}: {t}")
        for ax in scatter_axs[i]:
            ax.plot(*targets[i].T, label="Target", marker="*")
            ax.set_zlim([min(targets[i][:,2]) - 5, max(targets[i][:,2]) + 5])
            ax.set_aspect('equal')
            # ax.set_title(f"Target: {t}")

    # print("")

    # Results are keyed by run-log index because filesystem order is not stable.
    mae_column = pd.Series(index=df_runs.index, dtype=float)
    ae_column = pd.Series(index=df_runs.index, dtype=object)
    drops = 0
    in_df = 0
    for i, rf in enumerate(runs_files):
        target_idx = tests.index(runs["traj command"][rf[1]])
        model_idx = models.index(df_runs["inv_kin_model"][rf[1]])

        traj, ref_T, pos = parse_dataset(rf[0])
        _, pos_base, _ = polaris2base(traj, ref_T, pos)

        target_len_diff = len(pos_base) - len(targets[target_idx])
        if target_len_diff < 0:
            print("bad_dataset", i)
            mae_column.at[rf[1]] = np.nan
            ae_column.at[rf[1]] = np.array([])
            drops += 1
            continue
        abs_err_norm = np.linalg.norm(pos_base[target_len_diff:] - targets[target_idx], axis=1)
        error_dot = [np.dot((pos_base[i] - pos_base[i-target_len_diff])  / np.linalg.norm(pos_base[i] - pos_base[i-target_len_diff]), pos_base[i] - targets[target_idx][i-target_len_diff]) for i in range(target_len_diff, len(pos_base))] #  / np.linalg.norm((pos_base[i] - targets[target_idx][i-1])[i])
        pos_base = pos_base[target_len_diff:]
        
        mae = np.mean(abs_err_norm, where=~np.isnan(abs_err_norm))

        if mae == np.nan:
            print(rf[0], '\n')
        else:
            print(f"{runs['inv_kin_model'][rf[1]]}, test {tests.index(runs['traj command'][rf[1]])+1}, pt={runs['pause_time'][rf[1]]}s, len: {len(pos_base)+target_len_diff}")
            print(f'Mean Absolute error to Target: {mae:.4f}mm, {(abs_err_norm > 20).sum()} above plot limit\n')
        mae_column.at[rf[1]] = mae
        in_df += 1
        ae_column.at[rf[1]] = abs_err_norm

        scatter_axs[target_idx][model_idx].plot(*pos_base.T, label=f"pt={runs['pause_time'][rf[1]]}s", marker='.') # {runs['inv_kin_model'][rf[1]].split('_')[1]},
        error_axs[target_idx][model_idx+len(models)].plot(error_dot, '--', label=f"error_dot_pos_diff, pt={runs['pause_time'][rf[1]]}s") # {runs['inv_kin_model'][rf[1]].split('_')[1]},
        error_axs[target_idx][model_idx].plot(abs_err_norm, label=f"abs_error, pt={runs['pause_time'][rf[1]]}s") # {runs['inv_kin_model'][rf[1]].split('_')[1]},

        if scatter_axs[target_idx][model_idx].get_title() == "":
            scatter_axs[target_idx][model_idx].set_title(df_runs["inv_kin_model"][rf[1]])
        
        if error_axs[target_idx][model_idx].get_title() == "":
            error_axs[target_idx][model_idx].set_title(df_runs["inv_kin_model"][rf[1]])

    for err in error_axs:
        for i, ax in enumerate(err):
            ax.legend(title="Pause Time (pt)")
            if i < len(err) / 2:
                ax.set_ylim(0, y_lim)

    for scat in scatter_axs:
        for ax in scat:
            ax.legend(title="Pause Time (pt)")
            # if zlims != (0,0):
            #     ax.set_zlim([*zlims])

    def concat_lists(x):
        y = []
        for xx in x:
            y += list(xx[~np.isnan(xx)])
        if y == []:
            print("no data available")
        return np.array(y)


    # print(df_runs.shape)
    # print(df_runs.iloc[17:])
    print("in_df", in_df, "; drops", drops)
    df_runs["mae"] = mae_column
    df_runs["ae_array"] = ae_column
    bar_width = 0.75
    print("Box Plots")
    for i, test in enumerate(tests):
        print(f"Test {i+1}")
        # df_bar_plot = df_runs[df_runs["traj command"] == test].drop(columns=["traj command", "timestamp"])
        # df_box_plot = df_runs[df_runs["traj command"] == test].drop(columns=["traj command", "timestamp"]).pivot_table(
            # columns='pause_time', index='inv_kin_model', values='ae_array', aggfunc=concat_lists)
        df_box_plot = df_runs[df_runs["traj command"] == test].loc[:, ['ae_array', 'inv_kin_model', 'pause_time']]

        df_box_plot['inv_kin_model'] = pd.Categorical(df_box_plot['inv_kin_model'], categories=models, ordered=True)
        df_box_plot = df_box_plot.sort_values('inv_kin_model')
        df_box_plot = df_box_plot.explode('ae_array', True)
        # print(df_box_plot.shape)
        sns.boxplot(x='inv_kin_model', y='ae_array', hue='pause_time', data=df_box_plot, ax=bar_axs[i])
        for model in models:
            # print(df_box_plot[(df_box_plot['inv_kin_model'] == model) & (df_box_plot['ae_array'] > y_lim / 2)].shape)
            if (outside_num := df_box_plot[(df_box_plot['inv_kin_model'] == model) & (df_box_plot['ae_array'] > y_lim)].shape[0]) > 0:
                print("Values outside plot for model", model, ":", outside_num)
        bar_axs[i].set_ylim([0, y_lim])
        bar_axs[i].set_title(f"Test {i+1}")

    if save_plot:
        filenames = [run_folder + '_bars'] + [run_folder + '_scatter_t' + str(i) for i in range(len(tests))] + [run_folder + '_error_t' + str(i) for i in range(len(tests))]
        for i, fig in enumerate(figs):
            fig.savefig(os.path.join(run_folder, filenames[i] + '.pgf'))


def cont_validation_analysis(run_log_path="", run_folder="", zlims=(0,0), robot=False, save_plot=True):

    global tex_plots

    if run_folder == "":
        run_folder = "./val/"
    if run_log_path == "":
        run_log_path = os.path.join(run_folder, "run_log.txt")

    if not os.path.exists(run_log_path): return

    runs = parse_log(run_log_path)
    df_runs = pd.DataFrame(runs)
    print(run_folder)
    print(df_runs.shape)
    print(df_runs)
    # sys.exit()
    # print(runs.keys())
    figs = []
    tests = list(dict.fromkeys(df_runs["traj command"]))
    df_runs["inv_kin_model"] = df_runs["inv_kin_model"].apply(display_model_name)
    models = [model for model in MODEL_ORDER
              if model in df_runs["inv_kin_model"].unique()]
    runs_files = [(os.path.join(run_folder, file), df_runs["timestamp"].to_list().index(file.split('_')[-1][:-4])) for file in os.listdir(run_folder) if file.split('_')[-1][:-4] in df_runs['timestamp'].to_list()]
    runs_idxs = list(range(df_runs.shape[0]))
    [runs_idxs.remove(idx) for _, idx in runs_files]
    if runs_idxs:
        print("Missing files timestamps\n")
        print(df_runs.loc[runs_idxs, "timestamp"])
        sys.exit()

    # figs = [plt.figure(figsize=(6*len(tests),6))] + [plt.figure(figsize=(6*len(models),6)) for _ in range(len(tests)*2)]
    # figs = [plt.figure(figsize=(7*len(tests),6))] + [plt.figure(figsize=(6*len(models),6)) for _ in range(len(tests))] + [plt.figure(figsize=(6*len(models),4)) for _ in range(len(tests))]
    bar_figs = [plt.figure(figsize=(len(models)*1.15, 7)) for _ in range(len(tests))]
    scatter_figs = [plt.figure() for _ in range(len(models))]
    error_figs = [plt.figure() for _ in range(len(models))]
    for fig in error_figs:
        fig.tight_layout(pad=3)
    bar_axs = [fig.add_subplot(1, len(tests), i+1) for i, fig in enumerate(bar_figs)] # Add bar plot subplots
    scatter_axs = [fig.add_subplot(111, projection="3d") for fig in scatter_figs]
    error_axs = [fig.add_subplot() for fig in error_figs]
    # error_axs = [[fig.add_subplot(2, len(models), i+1) for i in range(len(models)*2)] for fig in figs[len(tests)+1:]]
    # print(error_axs[0])
    # print(len(error_axs[0]))
    y_lim = 10

    targets = []
    error_verts = []

    def angles(pts):
        vec1 = np.diff(pts[:-1], axis=0)
        # print(vec1.shape)
        vec2 = np.diff(pts[1:], axis=0)
        # dots = np.dot(vec1, vec2, axis)
        dots = np.sum(vec1 * vec2, axis=1)
        # print(dots.shape)
        norms = (np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1))
        angs = np.arccos(dots / norms)
        # print(angs.shape)

        return np.degrees(angs)

    if run_folder.strip('\\/')[-1].isdigit():
        if "square" not in run_folder:
            test_nr = " " + run_folder.strip('\\/')[-1]
        else:
            test_nr = " 1"
    else:
        test_nr = " 1"
    for i, t in enumerate(tests):
        targets.append(eval(t))
        print(f"Test {i+1}: {t}, len: {len(targets[i])} points")
        # scatter_figs[i].suptitle(f"Measured Position Along Trajectory - Test: {t.split('_')[1].split('(')[0].capitalize() + test_nr}")
        # error_figs[i].suptitle(f"Absolute Error Along Trajectory - Test: {t.split('_')[1].split('(')[0].capitalize() + test_nr}")

        angs = angles(targets[i])
        # print(angs)
        # error_verts.append(np.arange(1, len(targets[i]))[angs > (np.pi / 3)])
        error_verts.append(np.arange(1, len(targets[i])-1)[angs > 60])
        for ax in scatter_axs:
            ax.plot(*targets[i].T, label="Target", marker="*")
            # ax.scatter(*targets[i][1:-1].T, c=angs)
            if y_lim > 0 and not robot:
                ax.set_zlim([min(targets[i][:,2]) - y_lim/2, max(targets[i][:,2]) + y_lim/2])
            ax.set_aspect('equal')
            # ax.legend()
            if robot: draw_robot(ax, alpha_mult=0.8)

            # ax.set_title(f"Target: {t}")

    # print("")

    # return 0
    # Results are keyed by run-log index because filesystem order is not stable.
    mae_column = pd.Series(index=df_runs.index, dtype=float)
    ae_column = pd.Series(index=df_runs.index, dtype=object)
    drops = 0
    in_df = 0

    from scipy.spatial.distance import cdist
    def error_by_correspondance(source_traj, target_traj):
        distances = cdist(source_traj, target_traj, metric='euclidean')

        min_idxs = np.argmin(distances, axis=0)
    
        return min_idxs, np.linalg.norm(source_traj[min_idxs] - target_traj, axis=1)


    
    for i, rf in enumerate(runs_files):
        # print("file:", rf[0])
        target_idx = tests.index(df_runs["traj command"][rf[1]])
        model_idx = models.index(df_runs["inv_kin_model"][rf[1]])

        traj, ref_T, pos, count = parse_cont_dataset(rf[0])
        _, pos_base, _ = polaris2base(traj, ref_T, pos)

        target_len_diff = len(pos_base) // len(targets[target_idx])

        synced_idxs = [sum(count[:i])-1 for i in range(1, len(traj)+1)]
        synced_pos_base = pos_base[synced_idxs]
        
        try:
            assert len(synced_pos_base) == len(targets[target_idx])
        except Exception as e:
            print("len synced_pos_base != len targets;", e)
            mae_column.at[rf[1]] = np.nan
            ae_column.at[rf[1]] = np.array([])
            continue
        abs_err_norm = np.linalg.norm(synced_pos_base - targets[target_idx], axis=1)
        corresp_idxs, corresp_error = error_by_correspondance(pos_base, targets[target_idx])
        # error_dot = [np.dot((synced_pos_base[i] - synced_pos_base[i-1])  / np.linalg.norm(synced_pos_base[i] - synced_pos_base[i-1]), synced_pos_base[i] - targets[target_idx][i]) for i in range(1, len(synced_pos_base))]
        # pos_base = synced_pos_base
        
        mae = np.mean(abs_err_norm, where=~np.isnan(abs_err_norm))

        # if mae == np.nan:
        #     print(rf[0], '\n')
        # else:
        #     print(f"{df_runs['inv_kin_model'][rf[1]]}, test {tests.index(df_runs['traj command'][rf[1]])+1}, pt={df_runs['pause_time'][rf[1]]}s, len: {len(pos_base)+target_len_diff}")
        #     print(f'Mean Absolute error to Target: {mae:.4f}mm, {(abs_err_norm > 20).sum()} above plot limit\n')
        #     print(f'corresp Mean Absolute error to Target: {np.mean(corresp_error):.2f}mm, {(abs_err_norm > 20).sum()} above plot limit\n')
        mae_column.at[rf[1]] = mae
        in_df += 1
        ae_column.at[rf[1]] = corresp_error

        # scatter_axs[target_idx][model_idx].plot(*pos_base.T, label=f"pt={df_runs['pause_time'][rf[1]]}s") # {df_runs['inv_kin_model'][rf[1]].split('_')[1]},
        scatter_axs[model_idx].plot(*pos_base.T, label=f"Measured Trajectory") # {df_runs['inv_kin_model'][rf[1]].split('_')[1]},
        # scatter_axs[target_idx][model_idx].scatter(*synced_pos_base.T, color='k', s=8) # {df_runs['inv_kin_model'][rf[1]].split('_')[1]},
        # scatter_axs[target_idx][model_idx].scatter(*pos_base[corresp_idxs].T, color='k', s=8) # {df_runs['inv_kin_model'][rf[1]].split('_')[1]},
        
        # error_axs[target_idx][model_idx+len(models)].plot(error_dot, '--', label=f"error_dot_pos_diff, pt={df_runs['pause_time'][rf[1]]}s") # {df_runs['inv_kin_model'][rf[1]].split('_')[1]},
        # error_axs[target_idx][model_idx].plot(abs_err_norm, label=f"abs_error, pt={df_runs['pause_time'][rf[1]]}s") # {df_runs['inv_kin_model'][rf[1]].split('_')[1]},
        # error_axs[model_idx].plot(abs_err_norm, '--', label=f"Expected Position", alpha=0.7) # {df_runs['inv_kin_model'][rf[1]].split('_')[1]},

        # error_axs[target_idx][model_idx].plot(corresp_error, '--', color=error_axs[target_idx][model_idx].get_lines()[-1].get_color(), label=f"Best Correspondence Error, pt={df_runs['pause_time'][rf[1]]}s") # {df_runs['inv_kin_model'][rf[1]].split('_')[1]},
        # error_axs[model_idx].plot(corresp_error, color=error_axs[model_idx].get_lines()[-1].get_color(), label=f"Best Correspondence")
        error_axs[model_idx].plot(corresp_error, label=f"Absolute Error")
        error_axs[model_idx].plot([np.mean(corresp_error) for _ in range(len(corresp_error))], label="Mean Error")
        # {df_runs['inv_kin_model'][rf[1]].split('_')[1]}, # if you want pt

        if len(error_verts[target_idx]) > 0:
            error_axs[model_idx].vlines(error_verts[target_idx], [0] * len(error_verts[target_idx]), [y_lim] * len(error_verts[target_idx]),
            color='r', linestyle='dotted', alpha=0.8, label="Sharp Corners") # {df_runs['inv_kin_model'][rf[1]].split('_')[1]},
            
        if scatter_axs[model_idx].get_title() == "":
            scatter_axs[model_idx].set_title(df_runs['inv_kin_model'][rf[1]])
            # scatter_axs[target_idx][model_idx].set_title(f"Measured Position along Trajectory")
        
        if error_axs[model_idx].get_title() == "":
            error_axs[model_idx].set_title(df_runs['inv_kin_model'][rf[1]])

    # return
    for i, ax in enumerate(error_axs):
        # for i, ax in enumerate(err):
            # ax.legend(title="Pause Time (pt)")
            # if i < len(err) / 2:
            #     ax.set_ylabel(r"Absolute Error [mm]")
            # else:
            #     ax.set_ylabel(r"$AbsError \cdot NormPosDiff$ [mm]")
        # if i == 0: ax.legend()
        ax.legend(loc='upper right')
        # ax.legend(loc='lower left', bbox_to_anchor=(.8, .8))
        if y_lim > 0: ax.set_ylim(0, y_lim)
        ax.set_ylabel(r"Absolute Error [mm]")
        ax.set_xlabel(r"Trajectory Points")

    for ax in scatter_axs:
        # for ax in scat:
            # ax.legend(title="Pause Time (pt)")
        ax.legend()
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_zlabel('z [mm]')
            # if zlims != (0,0):
            #     ax.set_zlim([*zlims])

    def concat_lists(x):
        y = []
        for xx in x:
            y += list(xx[~np.isnan(xx)])
        if y == []:
            print("no data available")
        return np.array(y)


    # print(df_runs.shape)
    # print(df_runs.iloc[17:])
    print("in_df", in_df, "; drops", drops)
    df_runs["mae"] = mae_column
    df_runs["ae"] = ae_column
    bar_width = 0.75
    print("Box Plots")
    for i, test in enumerate(tests):
        # print(f"Test {i+1}")
        # df_bar_plot = df_runs[df_runs["traj command"] == test].drop(columns=["traj command", "timestamp"])
        # df_box_plot = df_runs[df_runs["traj command"] == test].drop(columns=["traj command", "timestamp"]).pivot_table(
            # columns='pause_time', index='inv_kin_model', values='ae', aggfunc=concat_lists)
        df_box_plot = df_runs[df_runs["traj command"] == test].loc[:, ['ae', 'inv_kin_model', 'pause_time']]

        df_box_plot['test'] = test
        df_box_plot['pause_time'] = df_box_plot['pause_time'].apply(lambda x: f'{x}s')
        df_box_plot['inv_kin_model'] = pd.Categorical(df_box_plot['inv_kin_model'], categories=models, ordered=True)
        df_box_plot = df_box_plot.sort_values('inv_kin_model')

        # print(df_box_plot.head(), df_box_plot.columns)
        print(df_box_plot.set_index('inv_kin_model')['ae'].apply(lambda x: np.mean(x)))
        print('\n'.join(map(str, ((i, row['inv_kin_model'], np.mean(row['ae']).astype(float)) for i, row in df_box_plot.iterrows()))))

        df_box_plot = df_box_plot.explode('ae', False)
        # print(df_box_plot.head())
        # print(df_box_plot.groupby('inv_kin_model')['ae'].mean().reset_index())
        # print(df_box_plot.shape)
        sns.boxplot(x='inv_kin_model', y='ae', data=df_box_plot, ax=bar_axs[i], color='paleturquoise',
            showmeans=True, meanline=True, meanprops={'color': 'blue', 'linewidth': 1.5}, medianprops={'linewidth': 1.5})
        for model in models:
            # print(df_box_plot[(df_box_plot['inv_kin_model'] == model) & (df_box_plot['ae'] > y_lim / 2)].shape)
            if (outside_num := df_box_plot[(df_box_plot['inv_kin_model'] == model) & (df_box_plot['ae'] > np.max(df_box_plot["ae"]) * 1.05)].shape[0]) > 0:
                print("Values outside plot for model", model, ":", outside_num)
                # print(melted_df[(melted_df['inv_kin_model'] == model) & (melted_df['value'] > y_lim)]) #  & (melted_df['pause_time'] == y)
        bar_axs[i].set_ylim([0, min(np.max(df_box_plot["ae"]) * 1.05, 10)])
        bar_axs[i].set_title(f"Average Error for each IK Model - Test: {test.split('_')[1].split('(')[0].capitalize() + test_nr}")
        bar_axs[i].set_xlabel("Inverse Kinematics Model")
        bar_axs[i].set_ylabel("Absolute Error [mm]")
        bar_axs[i].plot([], [], '--', linewidth=1, color='blue', label='Mean')
        bar_axs[i].plot([], [], '-', linewidth=1, color='gray', label='Median')
        bar_axs[i].scatter([], [], facecolor='none', edgecolor='gray', label='Outliers')
        bar_axs[i].grid(axis='y')
        bar_axs[i].tick_params(axis='x', rotation=45)
        # bar_axs[i].legend(title="Pause Time")
        bar_axs[i].legend(loc='lower left', bbox_to_anchor=(0.95, 1.01))
        # bar_axs[i].get_legend().set_visible(False)

    if save_plot:
        if tex_plots: ext = '.pgf'
        else: ext = '.png'
        plots_folder = os.path.join(run_folder, (os.path.basename(run_folder.strip('\\/')) + "_plots_split/"))
        print("saving plots to", plots_folder)
        if not os.path.exists(plots_folder): os.mkdir(plots_folder)
        filenames = [plots_folder + 'bars'] + \
                    [plots_folder + 'scatter_t' + str(i) for i in range(len(tests))] + \
                    [plots_folder + 'error-no-exp_t' + str(i) for i in range(len(tests))]
                    # [plots_folder + 'error_t' + str(i) for i in range(len(tests))]
        # for i, fig in enumerate(bar_figs + scatter_figs + error_figs):
        #     # if 'scatter_'
        #     fig.savefig(filenames[i] + '1.pgf')

        for i, fig in enumerate(bar_figs):
            # for label in fig.get_axes()[0].get_xticklabels():
            new_labels = [label.get_text().replace('PCC', 'CC') for label in fig.get_axes()[0].get_xticklabels()]
            fig.get_axes()[0].set_xticklabels(new_labels)
            print(fig.get_axes()[0].get_xticklabels())
            # fig.get_axes()[0].set_xticklabels([label.set_text(label.get_text().replace('PCC', 'CC')) for label in fig.get_axes()[0].get_xticklabels() if 'PCC' in label])
            fig.savefig(filenames[0] + '_t' + str(i + 1) + ext)
            # fig.savefig(filenames[0] + ext)

        for i, fig in enumerate(scatter_figs):
            if 'PCC' in (old_title := fig.get_axes()[0].get_title()):
                fig.get_axes()[0].set_title(old_title.replace('PCC', 'CC'))
            fig.savefig(filenames[1] + '_' + fig.get_axes()[0].get_title().lower() + ext)

        for i, fig in enumerate(error_figs):
            if 'PCC' in (old_title := fig.get_axes()[0].get_title()):
                fig.get_axes()[0].set_title(old_title.replace('PCC', 'CC'))
            fig.savefig(filenames[2] + '_' + fig.get_axes()[0].get_title().lower() + ext)

    # plt.close('all')
    return df_box_plot
