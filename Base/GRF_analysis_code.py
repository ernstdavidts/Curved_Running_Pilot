from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

from .Filter import butter_lowpass_filter, dual_butterworth
import os

print(os.getcwd())

def get_GRF_data(mat_path, force_plate, fs=1000):
    
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    qtmkey = list(data.keys())[-1]

    qtm = data[qtmkey]

    fp = qtm.Force[0] if force_plate == "FP1" else qtm.Force[1]

    Fx = -(fp.Force[0])
    Fy = -(fp.Force[1])
    Fz = -(fp.Force[2])
    
    # correction + filter
    correction = np.median(Fz[Fz<50])
    Fx = Fx - correction
    Fy = Fy - correction
    Fz = Fz - correction

    Fx = butter_lowpass_filter(Fx, cutoff= 80, fs=fs, order= 4)
    Fy = butter_lowpass_filter(Fy, cutoff= 80, fs=fs, order= 4)
    Fz = butter_lowpass_filter(Fz, cutoff= 80, fs=fs, order= 4)
    
    Fz[Fz < 35] = 0
    return [Fx, Fy, Fz]

def GRF_segm_ct(GRF_list, fs):
    Fx, Fy, Fz = GRF_list

    # Make control lines to let it work all the time like when there is no clear IC (when in 25_Curve_T1)
    IC = [i for i in range(0, len(Fz)) if Fz[i] != 0 and (Fz[i-1] == 0)][0]
    TO = [i for i in range(0, len(Fz)) if Fz[i] == 0 and (Fz[i-1] != 0)][0]

    segment_Fz = Fz[IC:TO]
    segment_Fy = Fy[IC:TO]
    segment_Fx = Fx[IC:TO]
    contact_time = (TO - IC)/fs

    return [segment_Fx, segment_Fy, segment_Fz], contact_time, IC, TO

def GRF_stats(segments):

    VILR = np.max(np.diff(segments[2]))
    
    return {
        "max_Fz": np.max(segments[2]),
        "VILR": VILR,
        "PBF": np.max(segments[1]),
        "max_Fx": np.max(np.abs(segments[0]))
    }

def analyze_folder_stats(folder):
    results = []
    plates = ["FP1","FP2"]
    folders = ["Curve", "Straight"]

    for condition in folders:
        condition_path = os.path.join(folder, condition)

        if not os.path.isdir(condition_path):
            continue

        for file in os.listdir(condition_path):
            if not file.endswith(".mat"):
                continue

            path = os.path.join(condition_path, file)

            participant = "02" if "MN" in path else "01"
            fs = 1000 if participant == "02" else 300
            
            if "55" in file:
                shoe = 55
            elif "45" in file:
                shoe = 45
            else:
                shoe = 25
            
            for plate in plates:

                condition_name = condition.lower()
                for plate in plates:
                    GRF_list = get_GRF_data(path, plate, fs=fs)
                    segments, _, IC,TO = GRF_segm_ct(GRF_list, fs=fs)
                    stats = GRF_stats(segments)

                    stats["participant"] = participant
                    stats["shoe"] = shoe
                    stats["condition"] = condition_name
                    stats["file"] = file
                    stats["plate"] = plate
                    stats["IC_time"] = (IC/fs)
                    stats["TO_time"] = (TO/fs)

                    results.append(stats)

    return pd.DataFrame(results)


def analyze_folder_segm(folder):
    results = []
    plates = ["FP1","FP2"]
    folders = ["Curve", "Straight"]

    for condition in folders:
        condition_path = os.path.join(folder, condition)

        if not os.path.isdir(condition_path):
            continue

        for file in os.listdir(condition_path):
            if not file.endswith(".mat"):
                continue
            
            parts = file.split("_")
            trial = parts[-1][0]
            path = os.path.join(condition_path, file)
            condition_name = condition.lower()
                
            participant = "02" if "MN" in path else "01"
            fs = 1000 if participant == "02" else 300
            
            if "55" in file:
                shoe = "55"
            elif "45" in file:
                shoe = "45"
            else:
                shoe = "25"
            
            for plate in plates:
                GRF_list = get_GRF_data(path, plate, fs=fs)
                segments, contact_time, IC, TO = GRF_segm_ct(GRF_list, fs=fs)
                
                results.append({
                    "participant": participant,
                    "shoe": shoe,
                    "contact time": contact_time,
                    "IC": IC/fs,
                    "TO": TO/fs,
                    "Fx": segments[0],
                    "Fy": segments[1],
                    "Fz": segments[2],
                    "trial": trial,
                    "plate": plate,
                    "condition": condition_name
                })
            
    return pd.DataFrame(results)


def segm_plot(df: pd.DataFrame, 
            force: str = "Fz", 
            participant: str=None, 
            condition:str=None,
            shoe: str=None, 
            plate: str=None, 
            trial: str =None,
            type: str=None,
            y_plot: str=None
    ):
    
    #Step 1: filter dataframe (if conditions are given)
    # Always filter your DataFrame first, then process it !
    if participant is not None:
        df = df[df["participant"] == participant]

    if shoe is not None:
        df = df[df["shoe"]== shoe]

    if plate is not None:
        df = df[df["plate"] == plate]

    if condition is not None:
        df = df[df["condition"] == condition]
    
    if trial is not None:
        df = df[df["trial"] == trial]

    # Step 2: plot
    plt.figure()
    
    if type == "interpolate":
        for _, row in df.iterrows():
            if row[force][0] > 1000:
                        print(f" trial {row["trial"]} deleted")
                        continue
            if y_plot == "normalised":
                if row["participant"] == "02":
                    y = np.asarray(row[force]/(72*9.81))
                elif row["participant"] == "01":
                    
                    y = np.asarray(row[force]/(81*9.81))
            else: 
                y = np.asarray(row[force])

            if len(y) < 10:
                print(
                    f"Skipped trial: Participant={row['participant']}, "
                    f"shoe={row['shoe']}, plate={row['plate']}, n={len(y)}"
                )
                continue

            x_orig = np.linspace(0, 1, len(y))
            x_new = np.linspace(0, 1, 100)
            f = interpolate.interp1d(x_orig, y)
            normalized_segment = f(x_new)
            
            plt.plot(normalized_segment)
            plt.title(f"normalised {force}")
            plt.xlabel("% of contact time")
            plt.ylabel(f"{force} (N)")
    else:
        for _, row in df.iterrows():
            if row[force][0] > 1000:
                        print(f" trial {row["trial"]} deleted")
                        continue
            if y_plot == "normalised":
                if row["participant"] == "02":
                    y = np.asarray(row[force]/(72*9.81))
                elif row["participant"] == "01":
                    y = np.asarray(row[force]/(81*9.81))
            else: 
                y = np.asarray(row[force])

            x = np.linspace(0, row["contact time"], len(y))
            plt.plot(x, y)
            plt.xlabel("contact time")
            plt.ylabel(f"{force} (N)")
            plt.title(f"{force} per contact time")
    
    plt.show()

def plot_stats(
    df: pd.DataFrame, 
    force: str ="Fz", 
    comparison: str ="shoe",
    participant: str = None,
    shoe: str = None,
    plate: str = None,
    condition: str = None, 
    y_plot: str =None
    ):
    
    plt.figure()
    x_new = np.linspace(0, 1, 100)

    if participant is not None:
        df = df[df["participant"] == participant]

    if shoe is not None:
        df = df[df["shoe"] == shoe]

    if plate is not None:
        df = df[df["plate"] == plate]

    if condition is not None:
        df = df[df["condition"] == condition]

    for comp, group_df in df.groupby(comparison):
        normalized_curves = []
        if comparison is None:
            comp="all"
        elif isinstance(comparison, str):
            comp = group_df[comparison].iloc[0]
        else:
            comp = " | ".join(str(group_df[col].iloc[0]) for col in comparison)
        
        for _, row in group_df.iterrows():    
            if row[force][0] > 1000:
                        print(f" trial {row["trial"]} deleted")
                        continue
            if y_plot == "normalised":
                if row["participant"] == "02":
                    y = np.asarray(row[force]/(72*9.81))
                elif row["participant"] == "01":
                    y = np.asarray(row[force]/(81*9.81))
            else: 
                y = np.asarray(row[force])

            if len(y) < 10:
                print(
                    f"Skipped trial: Participant={row['participant']}, "
                    f"shoe={row['shoe']}, plate={row['plate']}, n={len(y)}"
                )
                continue

            x_orig = np.linspace(0, 1, len(y))
            f = interpolate.interp1d(x_orig, y)
            normalized_y = f(x_new)
            
            normalized_curves.append(normalized_y)

        if not normalized_curves:
            continue

        curves_array = np.vstack(normalized_curves)
        mean_curve = np.mean(curves_array, axis=0)
        std_curve = np.std(curves_array, axis=0)

        line, = plt.plot(x_new * 100, mean_curve, label=f"{comp}")
        plt.fill_between(
            x_new * 100,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=line.get_color(),
            alpha=0.2,
            linewidth=0
        )

    plt.title(f"normalised mean {force}")
    plt.xlabel("% of contact time")
    plt.ylabel(f"Mean {force} curve (N)")
    plt.legend()
    plt.show()
