import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import hilbert, find_peaks


###############################################################################
# Helper Functions for Feature Computations
###############################################################################

def compute_signal_energy(signal):
    """
    Energy = sum(x[i]^2) over the entire signal.
    """
    return np.sum(signal ** 2)


def compute_mean_power(signal):
    """
    Mean Power = mean(x[i]^2).
    """
    return np.mean(signal ** 2)


def compute_snr(signal):



    signal_power = np.mean(signal ** 2)
    # approximate noise
    noise = np.std(signal - np.mean(signal))  # simplistic
    noise_power = noise ** 2 if noise != 0 else 1e-12
    snr_value = 10.0 * np.log10(signal_power / noise_power)
    return snr_value


def compute_mean_frequency(signal, fs=1000):


    n = len(signal)
    if n < 2:
        return np.nan

    # Real FFT
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # Power
    power = np.abs(fft_vals) ** 2
    total_power = np.sum(power)
    if total_power < 1e-12:
        return 0.0

    mean_freq = np.sum(freqs * power) / total_power
    return mean_freq


def compute_fractal_dimension(signal):


    # Petrosian’s fractal dimension (rough example)
    diff = np.diff(signal)
    N_delta = np.sum(diff[1:] * diff[:-1] < 0)
    n = len(signal)
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * N_delta)))


def compute_zero_crossings(signal):

    signs = np.sign(signal)
    zc = np.sum(np.diff(signs) != 0)
    return zc


def find_onset_time(signal, fs=1000, threshold=0.01):
    
    abs_signal = np.abs(signal)
    idx = np.where(abs_signal > threshold)[0]
    if len(idx) == 0:
        return np.nan
    return idx[0] / fs


###############################################################################
# Master Feature Computation for a Single Repetition
###############################################################################

def compute_emg_statistical_features(
        emg_array,
        fs=1000,
        muscle_name="deltoideus_anterior"
):
    """
    Compute a comprehensive set of features from the EMG signal for ONE repetition.
    Each feature is prefixed with <muscle_name> for clarity.

    Returns a dictionary: { "<muscle_name>_Mean": value, ... }
    """

    feats = {}
    N = len(emg_array)

    # Edge case: if repetition has no samples
    if N == 0:
        placeholders = [
            "Mean", "Median", "Mode", "StdDev", "Variance", "CV", "IQR", "Skewness", "Kurtosis",
            "Percentile_25", "Percentile_50", "Percentile_75", "Range_Q3_Q1", "Range_Q90_Q10",
            "Energy", "MeanPower", "TAV", "RMS", "Peak", "Min", "Range", "ARV", "ZeroCrossings",
            "MeanFrequency", "SNR", "Entropy", "FractalDimension", "CumulativeSum", "TimeToPeak",
            "OnsetTime"
        ]
        for p in placeholders:
            feats[f"{muscle_name}_{p}"] = np.nan
        return feats

    # Convert to numpy array, handle missing values
    x = np.array(emg_array, dtype=float)
    x = np.nan_to_num(x)  # if any NaNs, set to 0 or handle differently

    # 1) Central Tendency: Mean, Median, Mode
    mean_val = np.mean(x)
    feats[f"{muscle_name}_Mean"] = mean_val

    median_val = np.median(x)
    feats[f"{muscle_name}_Median"] = median_val


    hist, bin_edges = np.histogram(x, bins=50)
    mode_bin = np.argmax(hist)
    mode_est = 0.5 * (bin_edges[mode_bin] + bin_edges[mode_bin + 1])
    feats[f"{muscle_name}_Mode"] = mode_est

    # 2) Variability: StdDev, Variance, CV, IQR
    std_dev = np.std(x, ddof=1) if N > 1 else 0.0
    feats[f"{muscle_name}_StdDev"] = std_dev

    var_val = std_dev ** 2
    feats[f"{muscle_name}_Variance"] = var_val

    cv_val = (std_dev / mean_val) if np.abs(mean_val) > 1e-12 else np.nan
    feats[f"{muscle_name}_CV"] = cv_val

    p25 = np.percentile(x, 25)
    p75 = np.percentile(x, 75)
    iqr_val = p75 - p25
    feats[f"{muscle_name}_IQR"] = iqr_val

    # 3) Shape: Skewness, Kurtosis
    sk_val = skew(x, bias=False) if N > 1 else 0.0
    kt_val = kurtosis(x, bias=False) if N > 1 else 0.0
    feats[f"{muscle_name}_Skewness"] = sk_val
    feats[f"{muscle_name}_Kurtosis"] = kt_val

    # 4) Percentiles
    feats[f"{muscle_name}_Percentile_25"] = p25
    feats[f"{muscle_name}_Percentile_50"] = median_val  # same as median
    feats[f"{muscle_name}_Percentile_75"] = p75
    feats[f"{muscle_name}_Range_Q3_Q1"] = iqr_val  # same as IQR
    p10 = np.percentile(x, 10)
    p90 = np.percentile(x, 90)
    feats[f"{muscle_name}_Range_Q90_Q10"] = p90 - p10

    # 5) Energy and Power
    energy_val = compute_signal_energy(x)
    feats[f"{muscle_name}_Energy"] = energy_val

    mean_power = compute_mean_power(x)
    feats[f"{muscle_name}_MeanPower"] = mean_power

    # TAV = sum of abs(x)
    tav_val = np.sum(np.abs(x))
    feats[f"{muscle_name}_TAV"] = tav_val

    # RMS
    rms_val = np.sqrt(np.mean(x ** 2))
    feats[f"{muscle_name}_RMS"] = rms_val

    # 6) Amplitude Features
    peak_val = np.max(x)
    min_val = np.min(x)
    feats[f"{muscle_name}_Peak"] = peak_val
    feats[f"{muscle_name}_Min"] = min_val
    feats[f"{muscle_name}_Range"] = peak_val - min_val

    # ARV = Average Rectified Value = mean of abs(x)
    arv_val = np.mean(np.abs(x))
    feats[f"{muscle_name}_ARV"] = arv_val

    # 7) Signal Stability
    # Zero Crossings
    zc_val = compute_zero_crossings(x)
    feats[f"{muscle_name}_ZeroCrossings"] = zc_val

    # Mean Frequency
    mf_val = compute_mean_frequency(x, fs)
    feats[f"{muscle_name}_MeanFrequency"] = mf_val

    # SNR
    snr_val = compute_snr(x)
    feats[f"{muscle_name}_SNR"] = snr_val

    # 8) Signal Complexity
    # Entropy (Shannon)
    hist_counts, bin_edges = np.histogram(x, bins=50, density=True)
    hist_counts = hist_counts + 1e-12  # avoid zero
    ent_val = entropy(hist_counts, base=2)  # Shannon entropy in bits
    feats[f"{muscle_name}_Entropy"] = ent_val

    # Fractal Dimension (simple approach)
    fd_val = compute_fractal_dimension(x)
    feats[f"{muscle_name}_FractalDimension"] = fd_val

    # 9) Temporal Features
    # Cumulative Sum
    cumsum_val = np.sum(x)  # final value of cumsum
    feats[f"{muscle_name}_CumulativeSum"] = cumsum_val

    # Time-to-Peak:
    idx_peak = np.argmax(x)
    time_to_peak = idx_peak / fs
    feats[f"{muscle_name}_TimeToPeak"] = time_to_peak

    # Onset Detection:
    onset_time = find_onset_time(x, fs=fs, threshold=0.01 * peak_val)  # e.g., 1% of peak
    feats[f"{muscle_name}_OnsetTime"] = onset_time

    return feats


###############################################################################
# File-Level and Consolidated Feature Extraction
###############################################################################

def extract_features_from_emg_file(csv_path, muscle_name, fs=1000):
    """
    Reads one CSV with columns [EMG, Repetition].
    Groups by Repetition and computes the required features
    via 'compute_emg_statistical_features'.

    Returns a DataFrame with:
      Subject, Repetition, <features...>
    """
    filename = os.path.basename(csv_path)

    # Attempt to parse subject ID from filename "Subject_1_emg_deltoideus_anterior.csv"
    subject_match = re.search(r"Subject_(\d+)", filename, re.IGNORECASE)
    subject_id = subject_match.group(1) if subject_match else "Unknown"

    df = pd.read_csv(csv_path)
    required_cols = {"EMG", "Repetition"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"File {csv_path} missing columns: {required_cols - set(df.columns)}.")

    # Group by Repetition
    grouped = df.groupby("Repetition")
    all_feats = []
    for rep_id, group_data in grouped:
        emg_array = group_data["EMG"].to_numpy(dtype=float)

        feats_dict = compute_emg_statistical_features(
            emg_array, fs=fs, muscle_name=muscle_name
        )
        # Add identification
        feats_dict["Subject"] = subject_id
        feats_dict["Repetition"] = rep_id

        all_feats.append(feats_dict)

    return pd.DataFrame(all_feats)


def extract_all_emg_statistical_features(
        base_directory="processed_data_35_i",
        muscle_folders=(
                "emg_deltoideus_anterior",
                "emg_deltoideus_posterior",
                # and other muscle folders
        ),
        fs=1000,
        output_csv="OutputCSVFiles/consolidated_statistical_emg_features.csv",
        metadata_file=None
):
    """
    Iterates over each muscle subfolder, extracts a broad set of
    statistical EMG features for each Repetition and Subject,
    and consolidates them into one CSV.

    Output columns:
      Subject, Repetition,
      <muscle>_Mean, <muscle>_Median, <muscle>_Mode, <muscle>_StdDev, ... etc.

    If metadata_file is provided, merges on 'Subject' for additional columns.
    """
    import os

    all_dfs = []

    for muscle_name in muscle_folders:
        muscle_path = os.path.join(base_directory, muscle_name)
        if not os.path.isdir(muscle_path):
            print(f"[WARN] Muscle folder '{muscle_path}' not found. Skipping.")
            continue

        csv_files = glob.glob(os.path.join(muscle_path, "*.csv"))
        if not csv_files:
            print(f"[WARN] No CSV files in {muscle_path}. Skipping muscle: {muscle_name}")
            continue

        for csv_path in csv_files:
            print(f"[INFO] Processing EMG file: {csv_path}")
            df_feats = extract_features_from_emg_file(
                csv_path, muscle_name=muscle_name, fs=fs
            )
            all_dfs.append(df_feats)

    if not all_dfs:
        print("[INFO] No EMG features extracted. Check directory structure/CSV files.")
        return

    # Combine all
    combined_df = pd.concat(all_dfs, axis=0, ignore_index=True)
    # Group by (Subject, Repetition) and merge columns
    combined_df.set_index(["Subject", "Repetition"], inplace=True)
    wide_df = combined_df.groupby(level=["Subject", "Repetition"]).first()
    wide_df.reset_index(inplace=True)

    # Optionally merge with metadata
    if metadata_file and os.path.exists(metadata_file):
        meta_df = pd.read_csv(metadata_file)
        wide_df = pd.merge(wide_df, meta_df, on="Subject", how="left")

    # Save final CSV
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    wide_df.to_csv(output_csv, index=False)
    print(f"[INFO] EMG statistical features saved to {output_csv}")
