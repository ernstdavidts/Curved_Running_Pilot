from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
from scipy import interpolate
print(os.getcwd())

def get_mat_data(mat_path, variable="angles"):
    # load data and labels directly from the OpenSim python .mat file
    data = loadmat(mat_path, 
        squeeze_me=True,
        struct_as_record=False)
    
    if variable == "angles":
        # get labels first
        trial_names = data["labels"].ANGLES_TABLE
        first_trial = trial_names._fieldnames[0]
        labels = getattr(trial_names, first_trial)
        
        # get values
        trial_data = data["ANGLES_TABLE"]
    
    elif variable == "forces":
        trial_names = data["labels"].GRF_TABLE
        first_trial = trial_names._fieldnames[0]
        labels = getattr(trial_names, first_trial)
        trial_data = data["GRF_TABLE"]

    trial_dict = {}

    for i in range(len(trial_data._fieldnames)):
        trial = trial_data._fieldnames[i]
        values = getattr(trial_data, trial)

        if len(labels) == values.shape[1]:
            columns = [lab[0] if isinstance(lab, np.ndarray) else lab for lab in labels]
        else:
            raise ValueError("number of labels do not equal to number of colums")

        trial_dict[trial[:-4]] = pd.DataFrame(values, columns=columns)

    return trial_dict

def analyze_folder_OS(folder):
    all_data = {}

    for file in os.listdir(folder):
        if not file.endswith("python.mat"):
            continue

        path = os.path.join(folder, file)
        
        trial_dict = get_mat_data(path, variable="angles")

        for trial_name, os_df in trial_dict.items():
            parts = trial_name.split("_")

            for col in os_df.columns:
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
                    "time": os_df["time"],
                    "value": os_df[col],
                    "side": side
                    })
                elif len(parts[0]) == 3:
                    mini_df = pd.DataFrame({
                    "participant": "ED",
                    "shoe": str(parts[0][1:]),
                    "condition": parts[1],
                    "trial": parts[2][-1],
                    "time": os_df["time"],
                    "value": os_df[col],
                    "side": side
                    })
                else: 
                    mini_df = pd.DataFrame({
                    "participant": "ED",
                    "shoe": "55",
                    "condition": parts[0],
                    "trial": parts[1][-1],
                    "time": os_df["time"],
                    "value": os_df[col],
                    "side": side
                    })

                if col not in all_data:
                    all_data[col] = [mini_df]
                else:
                    all_data[col].append(mini_df)
                
    all_data = {
        key: pd.concat(value, ignore_index=False) for key, value in all_data.items()}
            
    return all_data

def merge_left_contacts(os_df, df_segm):
    os_df["trial"] = os_df["trial"].astype(str)
    os_df["shoe"] = os_df["shoe"].astype(str)

    merged = os_df.merge(
        df_segm[["participant", "shoe", "condition", "trial", "IC", "TO"]],
        on=["participant", "shoe", "condition", "trial"],
        how="left")

    stance = merged[
        (merged["time"] >= merged["IC"]) &
        (merged["time"] <= merged["TO"])
    ].copy()

    return stance

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
        if joint == "ankle_angle" or joint == "knee_angle" or "hip" in joint:
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
    x_new = np.linspace(0, 1, 100)
        
    for _, group in groups:
        group = group.sort_values('time')
        y = group["value"].to_numpy()

        if len(y) < 10:
                print(
                f"Skipped trial: participant={group['participant'].iloc[0]}, "
                )
                continue

        x_orig = np.linspace(0, 1, len(group["time"]))
        f = interpolate.interp1d(x_orig, y, kind="linear")
        normalized_y = f(x_new)

        if comparison is None:
            label="all"
        elif isinstance(comparison, str):
            label = group[comparison].iloc[0]
        else:
            label = " | ".join(str(group[col].iloc[0]) for col in comparison)

        if label not in normalized_curves:
            normalized_curves[label] = []  
        
        normalized_curves[label].append(normalized_y)

    plt.figure()

    for label, curves in normalized_curves.items():
        curves_array = np.vstack(curves)
        mean_curve = np.mean(curves_array, axis=0)
        std_curve = np.std(curves_array, axis=0)

        line, = plt.plot(x_new * 100, mean_curve, label=label)
        plt.fill_between(
            x_new * 100,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=line.get_color(),
            alpha=0.2,
            linewidth=0
        )

    plt.xlabel("% contact time")
    plt.ylabel("angle (°)")
    plt.title(joint)
    plt.legend()
    
    plt.show()

def stat_angle(joint, stat="mean", trial="Curve_1", specification="Rotation"):
    result = get_mat_data(joint)

    func = getattr(np, stat)
    
    if len(result) == 3:
        _, y_l, y_r = result
        return func(y_l), func(y_r)
    else:
        _, y = result
        return func(y)
