#### TO DO below: 

# - fix with Amelie that the names are the same of all the files: easier! 
# - How to control for noise ==> curve_4 => noise after sec 6 (i think) -> how to fix it?
# - link segment time with angle time !! -> fix data qtm: naming + conversion to opensim: GRF included: easier to calculate contact time

from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
from scipy import interpolate
print(os.getcwd())

def get_angle_data(mat_path, angle_path= "../../files/angles_with_labels.mat"):
    #load data
    data = loadmat(mat_path)
    trial_angle = data["ANGLES_TABLE"][0, 0]
    
    # load angle labels
    data_labels = loadmat(angle_path)
    raw_labels = data_labels["angle_labels"]
    labels = [str(x[0]) for x in raw_labels[0]]

    trial_dict = {}

    for name in trial_angle.dtype.names:
            trial_dict[name[:-4]] = pd.DataFrame(trial_angle[name], columns=labels)
    
    return trial_dict

def analyze_folder_OS(folder):
    all_data = {}

    for file in os.listdir(folder):
        if not file.endswith("python.mat"):
            continue

        path = os.path.join(folder, file)
        
        trial_dict = get_angle_data(path)

        for trial_name, angle_df in trial_dict.items():
            parts = trial_name.split("_")

            for col in angle_df.columns:
                if col == "time":
                    continue
                
                if col[-1] == "l":
                    side = "l"
                elif col[-1] == "r":
                    side = "r"
                else: 
                    side = None

                if parts[0] == "MN":
                    mini_df = pd.DataFrame({
                    "participant": parts[0],
                    "shoe": str(parts[1]),
                    "condition": parts[2],
                    "trial": parts[3],
                    "time": angle_df["time"],
                    "value": angle_df[col],
                    "side": side
                    })
                elif len(parts[0]) == 3:
                    mini_df = pd.DataFrame({
                    "participant": "ED",
                    "shoe": str(parts[0][1:]),
                    "condition": parts[1],
                    "trial": parts[2][-1],
                    "time": angle_df["time"],
                    "value": angle_df[col],
                    "side": side
                    })
                else: 
                    mini_df = pd.DataFrame({
                    "participant": "ED",
                    "shoe": "55",
                    "condition": parts[0],
                    "trial": parts[1][-1],
                    "time": angle_df["time"],
                    "value": angle_df[col],
                    "side": side
                    })

                if col not in all_data:
                    all_data[col] = [mini_df]
                else:
                    all_data[col].append(mini_df)
                
    all_data = {
        key: pd.concat(value, ignore_index=False) for key, value in all_data.items()}
            
    return all_data

def angle_plot(dict: dict, 
            joint: str = None, 
            participant: str=None, 
            condition: str = None,
            shoe: str=None, 
            trial: str=None, 
            side: str=None,
            comparison:str=None
    ):
    
    if side is None:
        if joint == "ankle" or joint == "knee" or joint == "hip":
            raise ValueError("No side of the joint was given")
        else:
            df = dict[joint] 
    else: 
        df = dict[f"{joint}_{side}"]
        df = df[df["side"] == side]

    if participant is not None:
        df = df[df["participant"]== participant]

    if shoe is not None:
        df = df[df["shoe"]== shoe]

    if condition is not None:
        df = df[df["condition"] == condition]

    if trial is not None:
        df = df[df["trial"] == trial]

    groups = df.groupby(["participant", "shoe", "trial", "condition"])
    normalized_curves = {}
        
    for _, group in groups:
        group = group.sort_values('time')

        y = group["value"].to_numpy()

        if len(y) < 10:
                print(
                f"Skipped trial: participant={group['participant'].iloc[0]}, "
                )
                continue

        x_orig = np.linspace(0, 1, len(group["time"]))
        x_new = np.linspace(0, 1, 100)
        
        f = interpolate.interp1d(x_orig, y, kind="linear")
        normalized_y = f(x_new)

        if comparison is None:
            label="all"
        else:
            label = group[comparison].iloc[0]

        if label not in normalized_curves:
            normalized_curves[label] = []  
        
        normalized_curves[label].append(normalized_y)

    for label, curves in normalized_curves.items():
        mean_curve = np.mean(curves, axis=0)
        plt.plot(mean_curve, label=str(label))

    plt.xlabel("% contact time ")
    plt.ylabel("angle (°)")
    plt.title(joint)
    plt.legend()
    
    plt.show()

def stat_angle(joint, stat="mean", trial="Curve_1", specification="Rotation"):
    result = get_angle_data(joint, trial, specification)

    func = getattr(np, stat)
    
    if len(result) == 3:
        _, y_l, y_r = result
        return func(y_l), func(y_r)
    else:
        _, y = result
        return func(y)
