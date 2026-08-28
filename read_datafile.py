#!/usr/bin/env python3
import numpy as np
import sys

# Set to True only when running locally with the optional visualization stack.
ENABLE_PLOTS = False

if ENABLE_PLOTS:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
from data_functions import (
    auto_curv_test_analysis_err,
    auto_curv_test_analysis_3d,
    auto_curv_test_analysis_2d,
    auto_vert_test_analysis,
    auto_basic_analysis,
    repeat_analysis,
    )

from val_plots_functions import (
    validation_analysis,
    cont_validation_analysis,
    )

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    # tkinter.Tk().withdraw()
    # filename = filedialog.askopenfilename(initialdir=os.getcwd() + '/data')

    # global filenames
    filenames = [
        # ("./data/dataset_6_2023-11-20T132506.txt", []),
        # ("./data/dataset_6_2023-11-20T135017.txt", []),
        # ("./data/dataset_1001_2023-11-20T154102.txt", []),
        # ("./data/dataset_101_2023-11-20T163508.txt", []),
        # ("./data/dataset_1001_2023-11-20T160038.txt", []),
        # ("./data/dataset_1001_2023-11-20T171434.txt", []), # servo error
        # ("./data/dataset_1001_2023-11-21T105942.txt", []),
        # ("./data/dataset_37_vert1_2023-11-21T162600.txt", []),
        # ("./data/dataset_55_vert1_2023-11-21T161719.txt", []),
        # ("./data/dataset_73_vert1_2023-11-21T161248.txt", []),
        # ("./data/dataset_181_vert2_2023-11-21T165321.txt", []),
        # ("./data/dataset_1001_2023-11-23T162547.txt", []),
        # ("./data/dataset_224_curv1_2023-11-27T151127.txt", []),
        # ("./data/dataset_976_curv1_2023-11-27T154255.txt", []), # 6 steps, 2 reps
        # ("./data/dataset_301_2023-11-27T164142.txt", []),
        # ("./data/dataset_180_vert2_2023-11-30T150240.txt", []),
        # ("./data/dataset_280_curv2_2023-12-07T183112.txt", []), # not complete
        # ("./data/dataset_284_curv2_2023-12-13T152932.txt", [800, 1220, 2000]),
        # ("./data/dataset_68_vert1_2023-12-19T145145.txt", []),
        # ("./data/dataset_35_curv2_2023-12-19T171459.txt", []),
        ("./data/dataset_272_curv2_2024-01-03T154745.txt", [800, 1220, 1800]),  # see len2jtheta@kinematics_functions
        # ("./data/dataset_273_curv2_2024-01-03T164457.txt", [800, 1220, 1800]),
        # ("./data/dataset_368_curv2_2024-06-19T114018.txt", [1000, 1220, 1450, 1700]), # TRASH
        # ("./data/dataset_272_curv2_2024-06-19T113039.txt", [1000, 1220, 1450, 1700]), # TRASH
        # ("./data/dataset_202_curv2_2024-06-19T140915.txt", [1000, 1220, 1600]), # TRASH
        # ("./data/dataset_204_grid_2024-01-12T162629.txt", []),
        # ("./data/dataset_205_2024-01-12T165908.txt", []),


        ("./data/dataset_1200_2023-12-13T161233.txt", []), # tese plot
        ("./data/dataset_1201_2023-12-19T155359.txt", []), # tese plot
        ("./data/dataset_1101_2024-02-09T103741.txt", []), # tese plot
        # ("./data/dataset_1200_2023-12-15T145833.txt", []), # many outliers
        # ("./data/dataset_1100_2024-02-07T165437.txt", []), # without extreme outlier
        # ("./data/dataset_1101_2024-02-07T165437.txt", []),
        # ("./data/dataset_1101_2024-02-09T141149.txt", []), # ~700

        # ("./data/dataset_10_check_2024-01-17T160944.txt", []),
        # ("./data/dataset_10_check_2024-02-07T113307.txt", []),
        # ("./data/dataset_10_check_2024-02-07T143908.txt", []),
        # ("./data/dataset_10_check_2024-02-07T165317.txt", []),
        # ("./data/dataset_10_check_2024-02-09T103153.txt", []),
        # ("./data/dataset_10_check_2024-02-09T162906.txt", []),
        # ("./data/dataset_10_check_2024-04-08T140528.txt", []),
        # ("./data/dataset_10_check_2024-04-08T143519.txt", []),
        # ("./data/dataset_10_check_2024-04-09T135715.txt", []),
        # ("./data/dataset_10_check_2024-06-19T104953.txt", []),
        # ("./data/dataset_10_check_2024-06-19T124855.txt", []),
        # ("./data/dataset_10_inv_fnn3_check_2024-04-09T155709.txt", []),
        # ("./data/dataset_10_inv_fnn3_check_2024-04-09T162259.txt", []),
        # ("./data/dataset_10_inv_pcc_check_2024-04-09T163120.txt", []),

        # ("./data/dataset_1051_rand3N_2024-02-07T114202.txt", []), # ~500
        # ("./data/dataset_1051_rand3N_2024-02-07T124945.txt", []),
        # ("./data/dataset_1049_rand3N_2024-02-07T144446.txt", []),
        # ("./data/dataset_1047_rand3N_2024-02-07T155230.txt", []), # ~900
        # ("./data/dataset_7687_rand3N_2024-02-09T113646.txt", []), # also aprox 1100
        # ("./data/dataset_1047_rand3N_2024-02-09T130942.txt", []),
        # ("./data/dataset_841_rand3N_2024-02-09T164215.txt", []),

        # ("./val/dataset_25_inv_fnn3_square_2024-04-09T164906.txt", []),
        # ("./val/dataset_33_inv_fnn3_square_2024-04-09T170307.txt", []),
        # ("./val/dataset_33_inv_pcc_square_2024-04-09T170525.txt", []),

        ]

    # auto_curv_test_analysis_err(filenames)
    # auto_curv_test_analysis_3d(filenames)
    # auto_curv_test_analysis_2d(filenames)
    # auto_vert_test_analysis(filenames)
    # auto_basic_analysis(filenames, concat=True)
    # repeat_analysis(filenames)
    # plt.show()
    # sys.exit()

    # validation_analysis(run_folder="./val/cont_circle/", save_plot=False) # zlims=(80,130)
    # validation_analysis(run_folder="./val/cont_circle2/", save_plot=False) # zlims=(80,130)
    # validation_analysis(run_folder="./val/circle/", save_plot=False) # zlims=(80,130)
    # validation_analysis(run_folder="./val/circle2/", save_plot=False) # zlims=(80,130)
    # validation_analysis(run_folder="./val/cont_square/", save_plot=False) # zlims=(80,130)
    # validation_analysis(run_folder="./val/square/", save_plot=False) # zlims=(80,130)

    # cont_validation_analysis(run_folder="./cont_val/_circle")
    # cont_validation_analysis(run_folder="./cont_val/_cont_circle")
    # cont_validation_analysis(run_folder="./cont_val/cont_circle", save_plot=False)
    df_circle = cont_validation_analysis(run_folder="./cont_val/cont_circle", save_plot=False, enable_plots=ENABLE_PLOTS)
    # cont_validation_analysis(run_folder="./cont_val/cont_circle2", robot=True)
    # cont_validation_analysis(run_folder="./cont_val/cont_circle3")
    # cont_validation_analysis(run_folder="./cont_val/cont_coil", save_plot=False, robot=False)
    # cont_validation_analysis(run_folder="./cont_val/cont_square2/", save_plot=False) # zlims=(80,130)
    df_coil = cont_validation_analysis(run_folder="./cont_val/cont_coil", save_plot=False, robot=False, enable_plots=ENABLE_PLOTS)
    df_square = cont_validation_analysis(run_folder="./cont_val/cont_square2/", save_plot=False, enable_plots=ENABLE_PLOTS) # zlims=(80,130)
    # cont_validation_analysis(run_folder="./cont_val/circle")
    # cont_validation_analysis(run_folder="./cont_val/circle2")
    # cont_validation_analysis(run_folder="./cont_val/circle3")
    # cont_validation_analysis(run_folder="./cont_val/coil")
    # cont_validation_analysis(run_folder="./cont_val/square2/", save_plot=False) # zlims=(80,130)

    # plt.show()
    # plt.close('all')
    sys.exit()

    df_full = pd.concat((df_circle, df_coil, df_square))  #
    print(df_full.head())
    print(df_full.groupby(['inv_kin_model', 'test'])['ae'].mean().reset_index())
    df_full = df_full.merge(df_full.groupby('test')['ae'].count().reset_index().rename({'ae': 'test_count'}, axis=1), on='test')
    print(df_full.head())
    # weighted_average = df_full.groupby('inv_kin_model').apply(lambda x: (x['ae_array'] * x['test_count']).sum() / x['test_count'].sum())
    weighted_average = df_full.groupby('inv_kin_model').apply(lambda x: (x['ae'] * x['test_count']).sum() / x['test_count'].sum())
    print(weighted_average)
    fig = plt.figure(figsize=(7.7, 6.5))
    ax = fig.add_subplot()
    # sns.boxplot(x='inv_kin_model', y='ae_array', data=df_full, ax=ax, color='paleturquoise',
                # showmeans=True, meanline=True, meanprops={'color': 'blue', 'linewidth': 1.5}, medianprops={'linewidth': 1.5})
    sns.boxplot(x='inv_kin_model', y='ae', data=df_full, ax=ax, color='paleturquoise',
                showmeans=True, meanline=True, meanprops={'color': 'blue', 'linewidth': 1.5}, medianprops={'linewidth': 1.5})
    # ax.set_ylim([0, min(np.max(df_full["ae_array"]) * 1.05, 10)])
    ax.set_ylim([0, min(np.max(df_full["ae"]) * 1.05, 10)])
    ax.set_title("Average Error for each IK Model - Test: All Trajectories")
    ax.set_xlabel("Inverse Kinematics Model")
    ax.set_ylabel("Absolute Error [mm]")
    ax.plot([], [], '--', linewidth=1, color='blue', label='Mean')
    ax.plot([], [], '-', linewidth=1, color='gray', label='Median')
    ax.scatter([], [], facecolor='none', edgecolor='gray', label='Outliers')
    ax.grid(axis='y')
    ax.tick_params(axis='x', rotation=45)
    # ax.legend(title="Pause Time")
    ax.legend(loc='lower left', bbox_to_anchor=(0.9, 0.95))
    new_labels = [label.get_text().replace('PCC', 'CC') for label in ax.get_xticklabels()]
    ax.set_xticklabels(new_labels)

    # Histogram of errors
    fig_hist = plt.figure(figsize=(10, 6))
    ax_hist = fig_hist.add_subplot()
    sns.histplot(data=df_full, x='ae', hue='inv_kin_model',
                 element="step", stat="density", common_norm=False, ax=ax_hist)
    ax_hist.set_title("Absolute Error Distribution by IK Model")
    ax_hist.set_xlabel("Absolute Error [mm]")
    ax_hist.set_ylabel("Density")
    ax_hist.set_xlim(0, 10)
    ax_hist.grid(axis='both', alpha=0.3)

    # Update legend labels for consistency if needed
    if ax_hist.get_legend():
        for text in ax_hist.get_legend().get_texts():
            text.set_text(text.get_text().replace('PCC', 'CC'))

    # Stacked Histogram (less crowded overlap)
    fig_stack = plt.figure(figsize=(10, 6))
    ax_stack = fig_stack.add_subplot()
    sns.histplot(data=df_full, x='ae', hue='inv_kin_model',
                 multiple="stack", bins=50, ax=ax_stack)
    ax_stack.set_title("Stacked Absolute Error Count by IK Model")
    ax_stack.set_xlabel("Absolute Error [mm]")
    ax_stack.set_xlim(0, 10)
    if ax_stack.get_legend():
        for text in ax_stack.get_legend().get_texts():
            text.set_text(text.get_text().replace('PCC', 'CC'))

    # ECDF Plot (Empirical Cumulative Distribution Function) - Cleanest for comparisons,
    fig_ecdf = plt.figure(figsize=(10, 6))
    ax_ecdf = fig_ecdf.add_subplot()
    sns.ecdfplot(data=df_full, x='ae', hue='inv_kin_model', ax=ax_ecdf)
    ax_ecdf.set_title("Cumulative Error Distribution")
    ax_ecdf.set_xlabel("Absolute Error [mm]")
    ax_ecdf.set_ylabel("Proportion of Points")
    ax_ecdf.set_xlim(0, 10)
    ax_ecdf.grid(True, which='both', alpha=0.3)
    if ax_ecdf.get_legend():
        ax_ecdf.get_legend().set_title('')
        for text in ax_ecdf.get_legend().get_texts():
            text.set_text(text.get_text().replace('PCC', 'CC'))

    # --- Professional Ridge Plot for Publication ---
    # Set paper-ready styling
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    # Scientific color palette (Viridis is grayscale-friendly)
    models = df_full['inv_kin_model'].unique()
    pal = sns.color_palette("viridis", n_colors=len(models))

    # Initialize the FacetGrid with a tighter aspect ratio for papers
    g = sns.FacetGrid(df_full, row="inv_kin_model", hue="inv_kin_model",
                      aspect=12, height=0.8, palette=pal)

    # Draw the densities with controlled transparency and clean outlines
    g.map(sns.kdeplot, "ae", bw_adjust=0.6, clip_on=False, fill=True, alpha=0.6, linewidth=1.2)
    g.map(sns.kdeplot, "ae", clip_on=False, color="black", lw=0.8, bw_adjust=0.6)

    # Add a vertical dashed line for the mean of each distribution
    def add_mean_line(x, color, label):
        ax = plt.gca()
        mean_val = x.mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.2, alpha=0.8)

    g.map(add_mean_line, "ae")

    # Draw the baseline
    g.refline(y=0, linewidth=1, linestyle="-", color='black', alpha=0.5, clip_on=False)

    # Improved labeling (aligned to the left of the peaks)
    def label(x, color, label):
        ax = plt.gca()
        clean_label = label.replace('PCC', 'CC')
        ax.text(-0.02, 0.2, clean_label, fontweight="bold", color='black',
                ha="right", va="center", transform=ax.transAxes)

    g.map(label, "ae")

    # Control overlap (hspace=0 would be no overlap, negative is more overlap)
    g.figure.subplots_adjust(hspace=-0.2)

    # Final cleanup of titles and axis
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=False, left=True)
    g.set_axis_labels("Absolute Error [mm]", "")

    for ax in g.axes.flat:
        ax.set_xlim(0, 10)
        ax.tick_params(axis='x', labelsize=11)
        # Add subtle x-grid for comparison
        ax.grid(axis='x', linestyle=':', alpha=0.4, color='gray')

    # --- Manually Construct Clean Legend ---
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # 1. Create handle for Mean Error (First)
    mean_handle = Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='Mean Error')

    # 2. Create handles for the IK Models (Using the palette colors)
    model_handles = [
        Patch(facecolor=pal[i], alpha=0.6, label=models[i].replace('PCC', 'CC'))
        for i in range(len(models))
    ]

    # Combine handles and labels
    all_handles = [mean_handle] + model_handles
    all_labels = [h.get_label() for h in all_handles]

    # Add the legend to the figure
    g.figure.legend(handles=all_handles, labels=all_labels,
                    loc='upper right', bbox_to_anchor=(0.95, 0.98), fontsize=11,
                    facecolor='white', framealpha=1)

    # plt.show()
