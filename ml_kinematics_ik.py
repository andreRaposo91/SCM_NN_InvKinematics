# ### Libraries

# Numerical libraries
import numpy as np

# Machine learning and data preprocessing libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Plotting library
import matplotlib.pyplot as plt

import os, sys, datetime
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# PCC model
from kinematics_functions import T_beModule
# from point_clouds import generate_random_points
# from add_error_to_kin import add_error_to_kin
# from ISTlogo_traj import ISTlogo_traj

# read data from Polaris dataset
from data_functions import parse_dataset, polaris2base

from nn_builder import DNNModelBuilder
from nn_builder import mlp_1 # , rnn

diff_flag = False
# diff_flag = True
diff_data = 'cable'
diff_data = 'xyz'

write_flag = False
# write_flag = True

# data_size_range = np.linspace(0.1, 1, 5)
# data_size_range = np.linspace(0.125, 1, 8)
data_size_range = [0.5]
test_split = 0.15

n_range = range(25, 60 + 1, 5)
h_range = range(1, 4 + 1)

n_range = range(20, 40 + 1, 10)
h_range = range(1, 2 + 1)

MAX_EPOCH = 2000
# MAX_EPOCH = 500
PATIENCE = 100
BATCH_SIZE = 128
ACTIVATION_FUN = 'selu'
ACTIVATION_FUN = 'tanh'
ACTIVATION_FUN = 'sigmoid'
ACTIVATION_FUN = 'elu'
ACTIVATION_FUN = 'relu'


if diff_flag:
    datasets = [
        "./data/dataset_1051_rand3N_2024-02-07T114202.txt", # ~500
        "./data/dataset_1051_rand3N_2024-02-07T124945.txt",
        "./data/dataset_1049_rand3N_2024-02-07T144446.txt",
        "./data/dataset_1047_rand3N_2024-02-07T155230.txt", # ~900
        "./data/dataset_7687_rand3N_2024-02-09T113646.txt", # also aprox 1100
        "./data/dataset_1047_rand3N_2024-02-09T130942.txt",
        "./data/dataset_841_rand3N_2024-02-09T164215.txt",
        "./data/dataset_1201_2023-12-19T155359.txt",
        ]
else:
    datasets = [
        "./data/dataset_1200_2023-12-13T161233.txt",
        "./data/dataset_1200_2023-12-15T145833.txt", # many outliers
        "./data/dataset_1201_2023-12-19T155359.txt",
        "./data/dataset_1100_2024-02-07T165437.txt", # without extreme outlier
        "./data/dataset_1101_2024-02-09T103741.txt",
        "./data/dataset_1101_2024-02-09T141149.txt", # ~700
        ]

datasets_result = []

for filename in datasets:
    traj, ref_T, pos = parse_dataset(filename)
    traj_, pos_base, est_pos_base = polaris2base(traj, ref_T, pos)
    datasets_result.append({"traj": traj, "traj_": traj_, "pos_base": pos_base, "est_pos_base": est_pos_base})

# for i, dataset in enumerate(datasets_result):
#     pos_err = dataset.get("pos_base") - dataset.get("est_pos_base")
#     abs_err = np.abs(pos_err)
#     mae = np.mean(np.sum(abs_err, axis=1))
#     mse = np.mean(np.linalg.norm(pos_err, axis=1))
#     print("\n", i)
#     print("MAE", mae)
#     print("MSE", mse)
#     print("RMSE", np.sqrt(mse))

# Polaris data preprocessing

x, y = [], []
if diff_flag:
    xyz_diff, cable_diff = [], []

jtheta2len = lambda p: (p - 4488.62157) / -26.03760

# Filtering
pcc_err_tol = 20
for i, dataset in enumerate(datasets_result):
    # rel_err_norm = np.linalg.norm(dataset.get("pos_base") - dataset.get("est_pos_base"), axis=1) / \
    #     np.linalg.norm(dataset.get("est_pos_base"), axis=1)
    # pcc_err_tol = np.mean(rel_err_norm) + np.std(rel_err_norm).astype(float) * 0.5
    # mask = rel_err_norm < pcc_err_tol
    # print(max(rel_err_norm))

    abs_err_norm = np.linalg.norm(dataset.get("pos_base") - dataset.get("est_pos_base"), axis=1)
    # pcc_err_tol = np.mean(abs_err_norm) + np.std(abs_err_norm).astype(float) * 3
    mask = abs_err_norm < pcc_err_tol

    x += list(dataset.get("pos_base")[mask])
    y += list(dataset.get("traj_")[mask])
    if diff_flag and diff_data == 'xyz':
        xyz_diff_temp = np.diff(np.insert(dataset.get("pos_base")[mask], 0, T_beModule(y[0], [], 0, 0), axis=0))
        xyz_diff += xyz_diff_temp / np.linalg.norm(xyz_diff_temp, axis=1)
    elif diff_flag and diff_data == 'cable':
        cable_diff_temp = np.diff(np.insert(dataset.get("traj_")[mask], 0, y[0], axis=0))
        cable_diff += cable_diff_temp / np.linalg.norm(cable_diff_temp, axis=1)

x_full = np.array(x)
y_full = np.array(y)

if diff_flag and diff_data == 'xyz':
    x_full = np.vstack((x_full, xyz_diff))
elif diff_flag and diff_data == 'cable':
    cable_diff = np.array(cable_diff)
    x_full = np.vstack((x_full, cable_diff))

# x_train_full, x_test_full, y_train_full, y_test_full = train_test_split(x_full, y_full, test_size=test_split, random_state=93216)
x_train_full, x_test, y_train_full, y_test = train_test_split(x_full, y_full, test_size=test_split, random_state=93216)

min_max_scaler = MinMaxScaler()
x_test = min_max_scaler.fit_transform(x_test) # pos

print(f'Train Set Size: {len(x_train_full)}')
print(f'Test Set Size: {len(x_test)}')

dt = datetime.datetime.now(datetime.timezone.utc).isoformat().split('.')[0].replace(':', '')
if write_flag:
    result_file = open(f'results/final_datasets/results_no_diff_{dt}.txt', 'w')
    if diff_flag: result_file = open(f'results/final_datasets/results_diff_{dt}.txt', 'w')
    # result_file.write(f'Train Split from Full Training Set: {data_size*100:.0f}%; Test Split: {test_split*100:.0f}%\n')
    result_file.write(f'Train Set Size: {len(x_train_full)}\n')
    result_file.write(f'Test Set Size: {len(x_test)}\n')
    result_file.write(f'NN Params: Max Epochs - {MAX_EPOCH}; Activation Function - \'{ACTIVATION_FUN}\'; Stopping Criteria Patience - {PATIENCE}\n')
    result_file.write(f'\nArchitecture;pct of train dataset;Epochs;MRE [%];MaxRE [%];MAE [mm];MaxAE [mm];MRE (Mean Length) [%];MaxRE (Mean Length) [%];MRE (Full-Scale) [%];MaxRE (Full-Scale) [%]')
    result_file.flush()

for data_size in data_size_range:
    if data_size != 1:
        x_train, _, y_train, _ = train_test_split(x_train_full, y_train_full, test_size=1-data_size)
    else: x_train, y_train = x_train_full, y_train_full
        
    # xyz_diff = min_max_scaler.fit_transform(np.array(xyz_diff))
    x_train = min_max_scaler.fit_transform(x_train) # pos


    # ## Model Design & Selection
    models = []
    for h in h_range:
        models =  [(mlp_1, [[n]*h], {'activation': ACTIVATION_FUN}) for n in n_range] # eg

    # Using the model builder
        builder = DNNModelBuilder(x_train, models)

            
            # result_file.close()
        # sys.exit()

        builder.train(x_train, y_train, epochs=MAX_EPOCH, batch_size=BATCH_SIZE, validation_split=test_split/(1 - test_split), patience=PATIENCE)
        # for i, model in enumerate(builder.architectures):
        #     print(f"Model {i+1}: {str(model[0]).split(' ')[1]}, {model[1][0]}")
        builder.compare_models(stack=True)
        # losses = builder.evaluate(x_test, y_test)
        # builder.save_models()

        pred = builder.predict(x_test)

        print("Mean Lengths:", np.mean(y_test))
        for i, p in enumerate(pred):
            # plot_positions_comparison(np.array(y_test), np.array(p))
            pred_abs_norm_err = np.abs(np.linalg.norm(p - y_test, axis=1))
            pred_rel_mean_err = pred_abs_norm_err / np.mean(y_test) * 100
            pred_rel_norm_err = np.abs(np.linalg.norm(p - y_test, axis=1)) / np.abs(np.linalg.norm(y_test, axis=1)) * 100
            pred_rel_scale_err = pred_abs_norm_err / (np.max(y_test[:,0]) - np.min(y_test[:,0])) * 100
            print(f"\nArchitechture: {str(builder.architectures[i][0]).split(' ')[1]}, {builder.architectures[i][1][0]}")
            if builder.early_stopping.stopped_epoch == None or builder.early_stopping.stopped_epoch == 0:
                print(f"{MAX_EPOCH};")
            else: print(f"stopped at {len(builder.histories[i].history['val_loss'])}; ")
            print(f"Mean Relative Error: {np.mean(pred_rel_norm_err):.3f}%")
            print(f"Max Relative Error: {max(pred_rel_norm_err):.3f}%")
            print(f"Mean Absolute Error: {np.mean(pred_abs_norm_err):.3f} mm")
            print(f"Max Absolute Error: {max(pred_abs_norm_err):.3f} mm")
            print(f"Mean Relative Error (Mean Length): {np.mean(pred_rel_mean_err):.3f} %")
            print(f"Max Relative Error (Mean Length): {max(pred_rel_mean_err):.3f} %")
            print(f"Mean Relative Error (Full-Scale): {np.mean(pred_rel_scale_err):.3f}%")
            print(f"Max Relative Error (Full-Scale): {max(pred_rel_scale_err):.3f}%")

            if write_flag:
                result_file.write(f"\n{str(builder.architectures[i][0]).split(' ')[1]}, {builder.architectures[i][1][0]}; ")
                result_file.write(f"{data_size*100:.2f}; ")
                if builder.early_stopping.stopped_epoch == None or builder.early_stopping.stopped_epoch == 0:
                    result_file.write(f"{MAX_EPOCH}; ")
                else: result_file.write(f"{len(builder.histories[i].history['val_loss'])}; ")
                result_file.write(f"{np.mean(pred_rel_norm_err):.3f}; ")
                result_file.write(f"{max(pred_rel_norm_err):.3f}; ")
                result_file.write(f"{np.mean(pred_abs_norm_err):.3f}; ")
                result_file.write(f"{max(pred_abs_norm_err):.3f}; ")
                result_file.write(f"{np.mean(pred_rel_mean_err):.3f}; ")
                result_file.write(f"{max(pred_rel_mean_err):.3f}; ")
                result_file.write(f"{np.mean(pred_rel_scale_err):.3f}; ")
                result_file.write(f"{max(pred_rel_scale_err):.3f}")
                result_file.flush()
        
        if write_flag: plt.savefig(f'./results/final_datasets/imgs/training_{data_size*100:.0f}_{dt}.png')


if write_flag: result_file.close()
else: plt.show()
# write_results()
