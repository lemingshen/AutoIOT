import os
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, resample
import argparse


def load_ecg_records(directory_path):
    ecg_records = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".hea"):
            record_name = os.path.splitext(filename)[0]
            record = wfdb.rdrecord(os.path.join(directory_path, record_name))
            annotation = wfdb.rdann(os.path.join(directory_path, record_name), "atr")
            ecg_signal = record.p_signal[:, 0]
            ecg_records[record_name] = (ecg_signal, record.fs, annotation)
    return ecg_records


def preprocess_ecg_signal(ecg_signal, fs, desired_fs=360):
    if fs != desired_fs:
        num_samples = int(len(ecg_signal) * desired_fs / fs)
        ecg_signal = resample(ecg_signal, num_samples)
    lowcut = 0.5
    highcut = 100
    nyquist = 0.5 * desired_fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype="band")
    filtered_ecg_signal = filtfilt(b, a, ecg_signal)
    return filtered_ecg_signal


def detect_r_peaks(ecg_signal, sampling_rate):
    diff_signal = np.diff(ecg_signal)
    squared_signal = diff_signal**2
    window_size = int(0.12 * sampling_rate)
    integrated_signal = np.convolve(
        squared_signal, np.ones(window_size) / window_size, mode="same"
    )
    refractory_period = int(0.2 * sampling_rate)
    peaks, _ = find_peaks(integrated_signal, distance=refractory_period)
    return peaks


def evaluate_detection_results(detected_peaks, annotations):
    true_positives = []
    false_positives = []
    false_negatives = []
    tolerance_window = int(0.1 * annotations.fs)
    annotation_indices = np.array(annotations.sample)
    for peak_index in detected_peaks:
        closest_annotation = np.argmin(np.abs(annotation_indices - peak_index))
        if (
            np.abs(annotation_indices[closest_annotation] - peak_index)
            <= tolerance_window
        ):
            true_positives.append(peak_index)
        else:
            false_positives.append(peak_index)
    for annotation_index in annotation_indices:
        if not np.any(np.abs(detected_peaks - annotation_index) <= tolerance_window):
            false_negatives.append(annotation_index)
    accuracy = (
        len(true_positives) / len(annotation_indices)
        if len(annotation_indices) > 0
        else 0
    )
    return accuracy, true_positives, false_positives, false_negatives


def output_results(record_name, detected_peaks, accuracy):
    r_peak_indices_list = detected_peaks.tolist()
    formatted_result = (
        f"Case {record_name}",
        f"R-peak indices: {np.array(r_peak_indices_list)}",
        f"Detection accuracy: {accuracy}",
    )
    return formatted_result


def main(input_file):
    ecg_records = load_ecg_records(input_file)
    for record_name, (ecg_signal, fs, annotation) in ecg_records.items():
        preprocessed_signal = preprocess_ecg_signal(ecg_signal, fs)
        detected_peaks = detect_r_peaks(preprocessed_signal, fs)
        accuracy, _, _, _ = evaluate_detection_results(detected_peaks, annotation)
        result_output = output_results(record_name, detected_peaks, accuracy)
        print(result_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ECG R-peak detection using MIT-BIH Arrhythmia Database."
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="Path to the dataset folder.",
    )
    args = parser.parse_args()

    main(args.input_file)
