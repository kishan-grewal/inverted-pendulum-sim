import numpy as np
from scipy.signal import butter, filtfilt

# Standard settling time parameters
SETTLING_PERCENT = 0.02  # 2% of initial deviation
SETTLING_DURATION = 1.0  # [s]
FILTER_CUTOFF_HZ = 5.0  # [Hz]


def calculate_settling_time(t, theta_deg, initial_deviation, 
                           disturbance_time=0.0,
                           settling_percent=SETTLING_PERCENT,
                           duration=SETTLING_DURATION,
                           cutoff_hz=FILTER_CUTOFF_HZ):
    """
    Standard 2% settling time with filtered signal.
    """
    # Low-pass filter
    dt = t[1] - t[0]
    fs = 1.0 / dt
    nyquist = fs / 2.0
    normal_cutoff = cutoff_hz / nyquist
    
    b, a = butter(2, normal_cutoff, btype='low', analog=False)
    theta_filtered = filtfilt(b, a, theta_deg)
    
    # Calculate tolerance
    tolerance = settling_percent * abs(initial_deviation)
    
    # Find settling time
    start_idx = np.searchsorted(t, disturbance_time)
    n_samples = int(duration / dt)
    
    theta_after = np.abs(theta_filtered[start_idx:])
    
    for i in range(len(theta_after) - n_samples):
        if np.all(theta_after[i:i+n_samples] < tolerance):
            return t[start_idx + i] - disturbance_time
    
    return None


def filter_angle_signal(t, theta_deg, cutoff_hz=FILTER_CUTOFF_HZ):
    """Low-pass filter angle measurements."""
    dt = t[1] - t[0]
    fs = 1.0 / dt
    nyquist = fs / 2.0
    normal_cutoff = cutoff_hz / nyquist
    
    b, a = butter(2, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, theta_deg)