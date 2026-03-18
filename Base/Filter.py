from scipy.signal import butter, sosfiltfilt
import numpy as np

def butter_lowpass_filter(
    data: list[float], cutoff: float = 100 , fs: float = 1000, order: int = 4) -> list[float]:
    """Pas een Butterworth low-pass filter toe en retourneer het gefilterde signaal.

    De functie gebruikt second-order sections (SOS) en `sosfiltfilt` voor
    zero-phase filtering (geen faseverschuiving).

    Args:
        data: 1D-iterable met numerieke waarden (list of numpy array).
        cutoff: afkapfrequentie in Hz.
        fs: samplefrequentie in Hz.
        order: filterorde (typisch 2-8).

    Returns:
        Numpy-array met de gefilterde waarden (zelfde lengte als input).
    """
    data = np.asarray(data)
    if data.size == 0:
        return data
    nyq = 0.5 * fs
    wn = float(cutoff) / nyq
    sos = butter(order, wn, btype='low', output='sos')
    return sosfiltfilt(sos, data)

def dual_butterworth(data, cutoff=100, fs=1000):
    data = butter_lowpass_filter(data, cutoff=cutoff, fs=fs, order=2)
    data = butter_lowpass_filter(data, cutoff=cutoff, fs=fs, order=2)
    return data

