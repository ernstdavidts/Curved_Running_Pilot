import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

import h5py


data = loadmat("Shoe25OpenSim.mat")
print(data.keys())

Angles_curve_1 = data["ANGLES_TABLE"][0,0]

Angles_curve_1.shape

data = loadmat("Shoe25OpenSim.mat")

# struct openen
angles = data["ANGLES_TABLE"][0,0]

# beschikbare trials
print(angles.dtype.names)

# één trial nemen
trial = angles["x25_Curve_1_mat"][0]

# data bekijken
print(trial)

time = trial[:,0]
hip_angle = trial[:,1]