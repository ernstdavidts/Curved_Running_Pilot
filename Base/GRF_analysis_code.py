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
    
    # Also for the other forces?
    Fz[Fz < 20] = 0
    return [Fx, Fy, Fz]

def GRF_segm_ct(GRF_list, fs):
    Fx, Fy, Fz = GRF_list

    # Make control lines to let it work all the time like when there is no clear IC (when in 25_Curve_T1)
    IC = [i for i in range(0, len(Fz)) if Fz[i] != 0 and (Fz[i-1] == 0)][0]
    TO = [i for i in range(0, len(Fz)) if Fz[i] == 0 and (Fz[i-1] != 0)][0]

    segment_Fz = Fz[IC:TO]
    segment_Fy = Fy[IC:TO]
    segment_Fx = Fx[IC:TO]
    contacttijd = (TO - IC)/fs

    return [segment_Fx, segment_Fy, segment_Fz], contacttijd

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

    for file in os.listdir(folder):

        if not file.endswith(".mat"):
            continue

        for plate in plates:
            path = os.path.join(folder, file)

            if "MN" in path:
                GRF_list = get_GRF_data(path, plate, fs=1000)
                segments, _ = GRF_segm_ct(GRF_list, fs=1000)
            else:
                GRF_list = get_GRF_data(path, plate, fs=300)
                segments, _ = GRF_segm_ct(GRF_list, fs=300)

            stats = GRF_stats(segments)

            if "55" in file:
                stats["shoe"] = 55
            elif "45" in file:
                stats["shoe"] = 45
            else:
                stats["shoe"] = 25

            if "MN" in file:
                stats["Participant"] = "02"
            else:
                stats["Participant"] = "01"
            stats["file"] = file
            stats["plate"] = plate
            results.append(stats)

    return pd.DataFrame(results)


def analyze_folder_segm(folder):
    results = []
    plates = ["FP1","FP2"]

    for file in os.listdir(folder):
        if not file.endswith(".mat"):
            continue

        path = os.path.join(folder, file)
        
        participant = "02" if "MN" in path else "01"
        fs = 1000 if participant == "02" else 300
        
        if "55" in file:
            shoe = "55"
        elif "45" in file:
            shoe = "45"
        else:
            shoe = "25"
        
        for plate in plates:
            GRF_list = get_GRF_data(path, plate)
            segments, contact_time = GRF_segm_ct(GRF_list, fs=fs)
            
            results.append({
                "Participant": participant,
                "shoe": shoe,
                "contact time": contact_time,
                "Fx": segments[0],
                "Fy": segments[1],
                "Fz": segments[2],
                "plate": plate
            })
            
    return pd.DataFrame(results)


def segm_plot(df: pd.DataFrame, 
            force: str = "Fz", 
            participant: str=None, 
            shoe: str=None, 
            plate: str=None, 
            type: str=None
    ):
    
    #Step 1: filter dataframe (if conditions are given)
    # Always filter your DataFrame first, then process it !
    if participant is not None:
        df = df[df["Participant"]== participant]

    if shoe is not None:
        df = df[df["shoe"]== shoe]

    if plate is not None:
        df = df[df["plate"] == plate]

    # Step 2: plot
    plt.figure()
    
    if type == "normalised":
        for _, row in df.iterrows():    
            x_orig = np.linspace(0, 1, len(row[force]))
            x_new = np.linspace(0, 1, 100)
            f = interpolate.interp1d(x_orig, row[force])
            normalized_segment = f(x_new)
            
            plt.plot(normalized_segment)
            plt.title(f"normalised {force}")
            plt.xlabel("% of contact time")
            plt.ylabel(f"{force} (N)")
    else:
        for _, row in df.iterrows():
            x = np.linspace(0, row["contact time"], len(row[force]))
            plt.plot(x, row[force])
            plt.xlabel("contact time")
            plt.ylabel(f"{force} (N)")
            plt.title(f"{force} per contact time")
    
    plt.show()
    