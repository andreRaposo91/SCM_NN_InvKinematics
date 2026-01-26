# Plots for comparison of True and Prediction Positions
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
