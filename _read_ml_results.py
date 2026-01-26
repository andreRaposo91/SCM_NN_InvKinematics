# tests mean and min, layers together, FNN

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

labels2 = f'Architechture;pct of train dataset;Epochs;MRE [%];MaxRE [%];MAE [mm];MaxAE [mm];MRE (Mean Length) [%];MaxRE (Mean Length) [%];MRE (Full-Scale) [%];MaxRE (Full-Scale) [%]'.split(';')

dfs = []

save_as_pgf = False
if save_as_pgf:
    import matplotlib
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'sans-serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

models = ["rand"]
folder = "./results/final_datasets3/"
# models_title = ["Random Points", "Random Points and Direction Vector", "Points w/ Nieghbours and Direction Vector"]
models_title = ["FNN3"]
result_files = ["./results/final_datasets2/" + file for file in os.listdir("./results/final_datasets2") if ".txt" in file and "_results" not in file] + \
    ["./results/final_datasets3/" + file for file in os.listdir("./results/final_datasets3") if ".txt" in file and "_results" not in file]

# models = ["diff_rand"]
# folder = "./results/final_datasets3/"
# models_title = ["FNN6"]
# result_files = ["./results/final_datasets3/results_diff_rand_2024-04-09T130207.txt",
#                 "./results/final_datasets3/results_diff_rand_2024-04-10T101624.txt",
#                 "./results/final_datasets3/results_diff_rand_2024-04-11T114825.txt",
#                 "./results/final_datasets3/results_diff_rand_2024-04-12T114550.txt",
#                 "./results/final_datasets3/results_diff_rand_2024-04-12T130323.txt",
#                 "./results/final_datasets3/results_diff_rand_2024-04-13T100152.txt",
#                 "./results/final_datasets3/results_diff_rand_2024-04-13T191150.txt",
#                 "./results/final_datasets3/results_diff_rand_2024-04-14T231148.txt",
#                 "./results/final_datasets3/results_diff_rand_2024-04-15T110531.txt"
#                 ]

# result_files = ["./results/final_datasets3/" + file for file in os.listdir("./results/final_datasets3") if ".txt" in file and "_results" not in file]

# for file in result_files:
    # print(file)
# for df in dfs:
#     df['Architechture'] = [tuple(map(int, arch.split('[')[1].strip(']').split(', '))) for arch in df['Architechture']]

#     df.insert(1, 'h', list(map(len, df['Architechture'])))
#     df.insert(1, 'n', [arch[0] for arch in df['Architechture']])

#     print(df.shape)

# mean_vmaxs = [0, 20, 10, 7.5, 7.5]
# min_vmaxs = [0, 18, 7.5, 5, 5]
mean_vmaxs = [0, 12, 12, 12, 12]
min_vmaxs = [0, 18, 7.5, 5, 5]

for i, model in enumerate(models):
    print("models:", model)
    # result_files2 = []
    result_files2 = [file for file in result_files if "./results/final_datasets2/" in file
           and ("results_" + model + "_2024") in file]
    
    result_files3 = [file for file in result_files if "./results/final_datasets3/" in file
           and ("results_" + model + "_2024") in file]
    
    print('\n'.join(result_files2), '\n', '\n'.join(result_files3))
    # plt.show()
    # print("results_" + model + "_2024")
    # for file in result_files:
    dfs = []
    dfs = [(file, pd.read_csv(file, header=3, sep=';', index_col=None)) for file in result_files2] + \
        [(file, pd.read_csv(file, header=3, sep=';', index_col=None)) for file in result_files3]
    for j, df_tup in enumerate(dfs):
        df = df_tup[1]
        # print(result_files[j])
        # print(df.iloc[0])
        # print(df.shape)
        # print(df.columns)
        try:
            df['Architechture'] = [tuple(map(int, arch.split('[')[1].strip(']').split(', '))) for arch in df['Architechture']]
            df.insert(1, 'h', list(map(len, df['Architechture'])))
            df.insert(1, 'n', [arch[0] for arch in df['Architechture']])
        except Exception as e:
            # print(df_tup[0])
            # print(df_tup[1])
            print("df processing failed:", e)
            # sys.exit()


    try:
        dfs_concat = pd.concat([el[1] for el in dfs], ignore_index=True, axis=0)
    except Exception as e:
        print("failed concat:", e)
        continue

    # mean_fig, mean_ax = plt.subplots(1, len(set(dfs_concat["h"])), figsize=(28, 4))
    mean_fig, mean_ax = plt.subplots(2, len(set(dfs_concat["h"]))//2, figsize=(16, 8))
    mean_ax = mean_ax.flatten()
    mean_fig.tight_layout(pad=3)
    mean_fig.suptitle(f"Mean MSE for {models_title[i]}")
    # min_fig, min_ax = plt.subplots(1, len(set(dfs_concat["h"])), figsize=(28, 4))
    min_fig, min_ax = plt.subplots(2, len(set(dfs_concat["h"]))//2, figsize=(16, 8))
    min_fig.tight_layout(pad=3)
    min_ax = min_ax.flatten()
    min_fig.suptitle(f"Minimum MSE for {models_title[i]}")
    for j, h in enumerate(set(dfs_concat["h"])):
        if h > 1: s = 's'
        else: s = ''
        mean_ax[j].set_title(f"{h} layer{s}")
        min_ax[j].set_title(f"{h} layer{s}")
        
        # try:
        heatmap_data = dfs_concat[dfs_concat['h'] == h].pivot_table(index='pct of train dataset', columns='n', values='MAE [mm]', aggfunc='mean')
        sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.2f', ax=mean_ax[j], vmin=0, vmax=mean_vmaxs[h], annot_kws={"size": 35 / np.sqrt(len(heatmap_data.columns))})
        min_idx = heatmap_data.stack().idxmin()
        row_idx, col_idx = min_idx
        mean_ax[j].add_patch(patches.Rectangle((heatmap_data.columns.get_loc(col_idx), heatmap_data.index.get_loc(row_idx)), 1, 1, fc='none', ec='lime', lw=3, clip_on=False))
        
        heatmap_data = dfs_concat[dfs_concat['h'] == h].pivot_table(index='pct of train dataset', columns='n', values='MAE [mm]', aggfunc='min')
        sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.2f', ax=min_ax[j], vmin=0, vmax=min_vmaxs[h], annot_kws={"size": 35 / np.sqrt(len(heatmap_data.columns))})
        min_idx = heatmap_data.stack().idxmin()
        row_idx, col_idx = min_idx
        min_ax[j].add_patch(patches.Rectangle((heatmap_data.columns.get_loc(col_idx), heatmap_data.index.get_loc(row_idx)), 1, 1, fc='none', ec='lime', lw=3, clip_on=False))
        # heatmap_data = dfs_concat[dfs_concat['h'] == h].pivot_table(index='pct of train dataset', columns='n', values='MAE [mm]', aggfunc='min', fill_value=True)
        # heatmap_data.replace(True, np.nan, inplace=True)
        # sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.2f', ax=ax[1], vmin=0, vmax=vmaxs[h])
        # min_idx = heatmap_data.stack().idxmin()
        # row_idx, col_idx = min_idx
        # ax[1].add_patch(patches.Rectangle((heatmap_data.columns.get_loc(col_idx), heatmap_data.index.get_loc(row_idx)), 1, 1, fc='none', ec='lime', lw=3, clip_on=False))
        
        # # heatmap_data = dfs_concat[dfs_concat['h'] == h].pivot_table(index='pct of train dataset', columns='n', values='MAE [mm]', aggfunc=np.argmin)
        # heatmap_data = dfs_concat[dfs_concat['h'] == h].pivot_table(index='pct of train dataset', columns='n', values='MAE [mm]', aggfunc='count')
        # # heatmap_data = dfs_concat[dfs_concat['h'] == h].pivot_table(index='pct of train dataset', columns='n', values='MAE [mm]', aggfunc='std')
        # # sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.0f', ax=ax[2], vmin=0)
        # sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.2f', ax=ax[2], vmin=0)
        # min_idx = heatmap_data.stack().idxmin()
        # row_idx, col_idx = min_idx
        # ax[2].add_patch(patches.Rectangle((heatmap_data.columns.get_loc(col_idx), heatmap_data.index.get_loc(row_idx)), 1, 1, fc='none', ec='lime', lw=3, clip_on=False))
        # except Exception as e:
        #     print("failed plots:", e)
        #     continue

        mean_ax[j].set_xlabel('n')
        mean_ax[j].set_ylabel(r'% of train dataset')
    # print(" ", file)

    if save_as_pgf:
        plots_folder = os.path.join(folder, f'{models_title[0].lower()}_plots')
        # print(plots_folder)
        if not os.path.exists(plots_folder): os.mkdir(plots_folder)
        # if isinstance(ax, plt.Axes):
        #     fig.savefig(plots_folder + f"/{models_title[0].lower()}_{h}_min_mae.pgf")
        # else:
        #     fig.savefig(plots_folder + f"/{models_title[0].lower()}_{h}_all.pgf")
        mean_fig.savefig(plots_folder + f"/{models_title[0].lower()}_mean_mea.pgf")
        min_fig.savefig(plots_folder + f"/{models_title[0].lower()}_min_mea.pgf")

plt.show()