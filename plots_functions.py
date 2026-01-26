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

from kinematics_functions import T_beModule
from point_clouds import generate_square, generate_circle, generate_coil
from draw_functions import draw_robot

