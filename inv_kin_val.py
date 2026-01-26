import numpy as np
import time

from kinematics_functions import invKspace_car, jtheta2len, len2jtheta
from data_functions import parse_dataset, polaris2base
from keras.models import load_model

def data_norm(data, comp=False):
    length = len(data[0])
    if comp: min_max_values = np.load('data_norm_w-diff_comp.npz')
    else: min_max_values = np.load('data_norm_w-diff.npz')
    min_values = min_max_values['min_values'][:length]
    max_values = min_max_values['max_values'][:length]
    return (data - min_values) / (max_values - min_values)

def fnn3_inv_kin(pts, model_file="", norm=True):
    if model_file == "":
        model_file = "./results/final_datasets3/models_2024-03-26T112415/mlp_1_62pct_3L_70.keras"

    model = load_model(model_file)
    if model.input_shape[1] != 3:
        print("Incorrect Model Type")
        return None

    start_time = time.time()
    if norm:
        predictions = model.predict(data_norm(pts))
    else: predictions = model.predict(pts)
    print('inference time:', time.time() - start_time)
    return predictions

def fnn6_inv_kin(pts, model_file="", norm=True):
    if model_file == "":
        model_file = "./results/final_datasets3/models_2024-04-09T130207/mlp_1_62pct_2L_80.keras"

    model = load_model(model_file)
    if model.input_shape[1] != 6:
        print("Incorrect Model Type")
        return None
    
    pos_diff = np.diff(pts, axis=0)
    pos_diff = np.insert(pos_diff / np.linalg.norm(pos_diff, axis=1, keepdims=True), 0, np.zeros((1, 3)), axis=0)
    pts_w_diff = np.concatenate((pts, pos_diff), axis=1)

    assert len(pts) == len(pts_w_diff)
    if norm: return model.predict(data_norm(pts_w_diff))
    else: return model.predict(pts_w_diff)

def fnn3_pcc_inv_kin(pts, model_file="", norm=True):
    if model_file == "":
        # model_file = "./results/final_datasets_comp/models_2024-04-30T233236/mlp_1_100pct_3L_50.keras"
        model_file = "./results/final_datasets_comp/models_2024-06-07T102829/mlp_1_87pct_4L_75.keras"


    model = load_model(model_file)
    if model.input_shape[1] != 3:
        print("Incorrect Model Type")
        return None

    pcc_est = [invKspace_car(*p, theta_flag=False) for p in pts]
    if norm: return model.predict(data_norm(pcc_est, comp=True))
    else: return model.predict(pcc_est)

def fnn6_pcc_inv_kin(pts, model_file="", norm=True):
    if model_file == "":
        # model_file = "./results/final_datasets_comp/models_2024-05-03T110939/mlp_1_75pct_4L_80.keras"
        model_file = "./results/final_datasets_comp/models_2024-06-06T121224/mlp_1_87pct_4L_70.keras"

    model = load_model(model_file)
    if model.input_shape[1] != 6:
        print("Incorrect Model Type")
        return None
    
    pcc_est = [invKspace_car(*p, theta_flag=False) for p in pts]
    pos_diff = np.diff(pcc_est, axis=0) # not really pos_diff, more like pcc diff
    pos_diff = np.insert(pos_diff / np.linalg.norm(pos_diff, axis=1, keepdims=True), 0, np.zeros((1, 3)), axis=0)
    pts_w_diff = np.concatenate((pcc_est, pos_diff), axis=1)

    assert len(pts) == len(pts_w_diff) and pts_w_diff.shape[1] == model.input_shape[1]
    if norm: return model.predict(data_norm(pts_w_diff, comp=True))
    else: return model.predict(pts_w_diff)

def rnn_inv_kin(pts, model_file="", norm=False):
    if model_file == "":
        model_file = "./results/final_datasets_rnn/models_2024-04-03T155223/rnn_100pct_2L_70.keras"

    model = load_model(model_file)
    if len(model.input_shape) != 3 or model.input_shape[2] != 3:
        print("Incorrect Model Type")
        return None
    
    num_time_steps = model.input_shape[1]
    if norm: pts = data_norm(pts)
    pts_rnn = np.array([tuple(pts[i:i+num_time_steps,:]) for i in range(0, len(pts) - num_time_steps + 1)])
    # print(pts_rnn.shape)
    # print(pts_rnn[:4])
    if not (len(pts) - num_time_steps + 1) == len(pts_rnn):
        print(pts_rnn[:5])
        raise AssertionError(f"{len(pts)} | {len(pts_rnn)}")

    return model.predict(pts_rnn)

def rnn_pcc_inv_kin(pts, model_file="", norm=False):
    if model_file == "":
        # model_file = "./results/final_datasets_rnn_comp/models_2024-05-07T105044/rnn_75pct_3L_50.keras"
        model_file = "./results/final_datasets_rnn_comp/models_2024-06-11T204342/rnn_25pct_2L_70.keras"

    # print(pts[:5])
    pcc_est = np.array([invKspace_car(*p, theta_flag=False) for p in pts])

    model = load_model(model_file)
    if len(model.input_shape) != 3 or model.input_shape[2] != 3:
        print("Incorrect Model Type")
        return None
    
    num_time_steps = model.input_shape[1]
    if norm: pcc_est = data_norm(pcc_est, comp=True)
    pts_rnn = np.array([tuple(pcc_est[i:i+num_time_steps,:]) for i in range(0, len(pts) - num_time_steps + 1)])
    print(pcc_est.shape)
    # print(pts_rnn.shape)
    # print(pcc_est[:4], pcc_est[-4:])
    # print(pts_rnn[:4], pts_rnn[-4:])
    if not (len(pts) - num_time_steps + 1) == len(pts_rnn):
        print(pts_rnn[:5])
        raise AssertionError(f"{len(pts)} | {len(pts_rnn)}")

    return model.predict(pts_rnn)
    
# if __name__ == "__main__":
#     datasets = [
#         "./data/dataset_1200_2023-12-13T161233.txt",
#         "./data/dataset_1200_2023-12-15T145833.txt", # many outliers
#         "./data/dataset_1201_2023-12-19T155359.txt",
#         "./data/dataset_1100_2024-02-07T165437.txt", # without extreme outlier
#         "./data/dataset_1101_2024-02-09T103741.txt",
#         "./data/dataset_1101_2024-02-09T141149.txt", # ~700
#     ]

#     jtheta2len = lambda p: (p - 4488.62157) / -26.03760

#     datasets_result = []

#     for filename in datasets:
#         traj, ref_T, pos = parse_dataset(filename)
#         traj_, pos_base, est_pos_base = polaris2base(traj, ref_T, pos)
#         datasets_result.append({"traj": traj, "traj_": traj_, "pos_base": pos_base, "est_pos_base": est_pos_base})

#     x, y = [], []
#     x_test, y_test = [], []

#     pcc_err_tol = 20
#     for i, dataset in enumerate(datasets_result):
#         abs_err_norm = np.linalg.norm(dataset.get("pos_base") - dataset.get("est_pos_base"), axis=1)
#         # pcc_err_tol = np.mean(abs_err_norm) + np.std(abs_err_norm).astype(float) * 3
#         mask = abs_err_norm < pcc_err_tol
#         print("mask", i, all(mask))
#         x += list(dataset.get("pos_base")[mask])
#         y += list(dataset.get("traj_")[mask])

#     pred = np.array([invKspace_car(*pt, theta_flag=False) for pt in x])
#     print("MSE:", np.mean(np.linalg.norm(pred - y, axis=1)))
