# Plots for comparison of True and Prediction Positions
import glob
from pathlib import Path
import typing
from collections import defaultdict

import plotly.graph_objects as go
import matplotlib.pyplot as plt

def write_results():
    
    return 0

def plot_positions_comparison(real_positions, predicted_positions):
    # Extracting x, y, z coordinates for real positions
    x_real = real_positions[:, 0]
    y_real = real_positions[:, 1]
    z_real = real_positions[:, 2]

    # Extracting x, y, z coordinates for predicted positions
    x_pred = predicted_positions[:, 0]
    y_pred = predicted_positions[:, 1]
    z_pred = predicted_positions[:, 2]

    # Creating the figure
    fig = go.Figure()

    fig.update_layout(
        autosize = False,
        width = 500,
        height = 500,
        )

    # Adding real positions trace
    fig.add_trace(go.Scatter3d(x=x_real, y=y_real, z=z_real,
                               mode='markers', 
                            #    mode='markers+lines', 
                               marker=dict(size=3, color='blue', symbol='circle'),
                               name='Real Positions'))

    # Adding predicted positions trace
    fig.add_trace(go.Scatter3d(x=x_pred, y=y_pred, z=z_pred,
                               mode='markers', 
                            #    mode='markers+lines', 
                               marker=dict(size=3, color='red', symbol='x'),
                               name='Predicted Positions'))

    # Update plot layout
    fig.update_layout(
        title='Real vs Predicted End Effector Positions',
        scene=dict(
            xaxis_title='Servo 1',
            yaxis_title='Servo 2',
            zaxis_title='Servo 3'
            # xaxis_title='X Axis',
            # yaxis_title='Y Axis',
            # zaxis_title='Z Axis'
        ),
        margin=dict(r=10, l=10, b=10, t=30)
    )

    # Show the plot
    fig.show()


def keras_model_sizes(search_folder, pattern=''):
    from keras.models import load_model
    models = glob.iglob(search_folder + pattern)
    model_sizes = {}
    ls, ns = set(), set()
    for model in models:
        model_path = Path(model)
        try:
            model_obj = load_model(model, safe_mode=False)
        except Exception:
            continue
        lr, nr = model_path.stem.split('_')[:-2]
        l = int(lr.rstrip('L'))
        n = int(nr)
        ls.add(l)
        ns.add(n)
        model_sizes[(l,n)] = model_obj.count_params()

    print(ls, ns)


def fnn_params_from_specs(d_in, widths):
    """
    Compute total parameter count for an MLP with Dense layers only.

    widths: list[int]
      Interpreted as hidden layer sizes, stacked in order.
      Example: [30, 30] means Dense(30) -> Dense(30) -> Dense(3)

    d_out is fixed to 3 because your RNN builder ends with Dense(3).
    Biases included (Keras Dense default).
    """
    def dense_params(a, b):
        return a * b + b  # weights + bias

    total = 0
    prev = d_in

    for w in widths:
        total += dense_params(prev, w)
        prev = w

    # output head Dense(3)
    total += dense_params(prev, 3)
    return total


def rnn_params_from_specs(d_in, specs):
    """
    Compute total parameter count for your rnn() builder given:
      - d_in: data.shape[2] (input feature dim)
      - specs: a list like [(units1, ret_seq1), ..., (unitsN, ret_seqN)]

    Assumes SimpleRNN_layer is tf.keras.layers.SimpleRNN with bias=True (default),
    and your builder ends with Dense(3).
    """
    # Keras SimpleRNN(units=h) params: d*h + h*h + h
    def simple_rnn_params(input_dim, h):
        return input_dim * h + h * h + h

    # Dense(a -> b) params: a*b + b
    def dense_params(a, b):
        return a * b + b

    # Match your builder logic
    # loop over specs[0:-1]
    ret_seq_flag = True
    total = 0

    # input_dim seen by first layer is d_in
    input_dim = d_in

    for units, ret_seq in specs[:-1]:
        if ret_seq_flag and ret_seq:
            # add SimpleRNN(return_sequences=True)
            total += simple_rnn_params(input_dim, units)
            # next SimpleRNN sees input_dim = units (feature dim = units)
            input_dim = units
        elif ret_seq_flag and (not ret_seq):
            # add SimpleRNN(return_sequences=False)
            total += simple_rnn_params(input_dim, units)
            ret_seq_flag = False
            # after returning a vector, Dense(units, ...) layers see input_dim = units
            input_dim = units
        else:
            # ret_seq_flag is already False => Dense(units, ...)
            total += dense_params(input_dim, units)
            input_dim = units

    # final layer from specs[-1]
    last_units, _ = specs[-1]
    if ret_seq_flag:
        # add SimpleRNN(return_sequences=False)
        total += simple_rnn_params(input_dim, last_units)
        input_dim = last_units
    else:
        # add Dense(last_units, ...)
        total += dense_params(input_dim, last_units)
        input_dim = last_units

    # final Dense(3)
    total += dense_params(input_dim, 3)
    return total


if __name__ == '__main__':
    fnn_layer_specs = []
    rnn_layer_specs = []
    ns = list(range(20, 90, 10))
    for n in ns:
        fnn_layer_specs.extend((
            [n] * i for i in range(1, 4 + 1)
        ))
        rnn_layer_specs.extend((
            [(n, False)],
            [(n, False)] * 2,
            [(n, True)] + [(n, False)] * 2,
            [(n, True)]  * 2 + [(n, False)] * 2,

        ))
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f'Parameter Counts for Neural Networks')
    for inputs in (3, 6):
        print(f'{inputs} input networks')
        fnn_params_dict = defaultdict(list)
        rnn_params_dict = defaultdict(list)
        curr_n = None
        for fnn, rnn in zip(fnn_layer_specs, rnn_layer_specs):
            fnn_params = fnn_params_from_specs(inputs, fnn)
            rnn_params = rnn_params_from_specs(inputs, rnn)
            fnn_params_dict[len(fnn)].append(fnn_params)
            rnn_params_dict[len(rnn)].append(rnn_params)
            assert fnn[0] == rnn[0][0]
            if curr_n != fnn[0]:
                curr_n = fnn[0]
                print(f'- {curr_n}N')
            print(f'  - {len(fnn)}L: FNN={fnn_params:6d} RNN={rnn_params:6d} RNN/FNN={rnn_params/fnn_params:.4f}')
        legend = []
        for l in range(1, 4+1):
            axs_idx = l - 1
            ax = axs[axs_idx // 2][axs_idx % 2]
            ax.plot(ns, fnn_params_dict[l])
            ax.plot(ns, rnn_params_dict[l])
            ax.set_title(f'{l} Layer{"s" if l > 1 else ""}')
            if inputs == 6:
                ax.legend(['FNN - 3 inputs', 'RNN - 3 inputs',
                           'FNN - 6 inputs', 'RNN - 6 inputs',])
    plt.show()
    # keras_model_sizes('results/', pattern='final_datasets_*/*/rnn*.keras')
