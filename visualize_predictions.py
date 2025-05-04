#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SSVEP Prediction Visualization

This script shows continuous prediction results for each dataset,
similar to how the simulator works, to better visualize classification performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import os
from ssvep_classifier import load_data, preprocess_eeg
from scipy import signal
import time

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)
os.makedirs('videos', exist_ok=True)

# Constants
fs = 256  # Sampling rate (Hz)
stim_freqs = [15, 12, 10, 9]  # Stimulus frequencies (top, right, bottom, left)
freq_names = ["15 Hz (Top)", "12 Hz (Right)", "10 Hz (Bottom)", "9 Hz (Left)"]

def create_reference_signals(freqs, fs, T):
    """Create reference signals for FBCCA"""
    t = np.arange(0, T, 1/fs)
    Y = {}
    
    for freq in freqs:
        signals = []
        
        # Add harmonics (sine and cosine)
        for h in range(1, 6):  # Use 5 harmonics
            signals.append(np.sin(2 * np.pi * h * freq * t))
            signals.append(np.cos(2 * np.pi * h * freq * t))
        
        Y[freq] = np.vstack(signals)
    
    return Y

def canonical_correlation(X, Y):
    """Calculate canonical correlation between two multivariate signals"""
    # Center the data
    X = X - X.mean(axis=1, keepdims=True)
    Y = Y - Y.mean(axis=1, keepdims=True)
    
    # Calculate covariance matrices
    Cxx = np.cov(X)
    Cyy = np.cov(Y)
    Cxy = np.cov(X, Y)[:X.shape[0], X.shape[0]:]
    
    # Add small regularization to avoid singularity issues
    Cxx += np.eye(Cxx.shape[0]) * 1e-8
    Cyy += np.eye(Cyy.shape[0]) * 1e-8
    
    # Compute canonical correlations
    Cxx_inv_sqrt = np.linalg.inv(np.linalg.cholesky(Cxx))
    Cyy_inv_sqrt = np.linalg.inv(np.linalg.cholesky(Cyy))
    
    T = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt
    
    # SVD to get correlations
    U, s, Vh = np.linalg.svd(T, full_matrices=False)
    
    # Return maximum correlation
    return s[0]

def classify_epoch(eeg_epoch, stim_freqs, fs=256):
    """Classify an epoch of EEG data using FBCCA"""
    # Preprocess the epoch
    epoch_duration = eeg_epoch.shape[1] / fs
    
    # Create reference signals
    Y_ref = create_reference_signals(stim_freqs, fs, epoch_duration)
    
    # Calculate CCA for each frequency
    corr_values = []
    for freq in stim_freqs:
        corr = canonical_correlation(eeg_epoch, Y_ref[freq])
        corr_values.append(corr)
    
    # Get the frequency with highest correlation
    predicted_idx = np.argmax(corr_values)
    predicted_freq = stim_freqs[predicted_idx]
    confidence = corr_values[predicted_idx] / max(sum(corr_values), 1e-10)
    
    return predicted_freq, confidence, corr_values

def find_trigger_intervals(trigger_channel):
    """Find start and end points of each trigger in the data"""
    # Find trigger onsets (0->1 transitions)
    trigger_onsets = np.where(np.diff(np.concatenate(([0], trigger_channel))) == 1)[0]
    # Find trigger offsets (1->0 transitions)
    trigger_offsets = np.where(np.diff(np.concatenate((trigger_channel, [0]))) == -1)[0]
    
    # Ensure equal number of onsets and offsets
    min_len = min(len(trigger_onsets), len(trigger_offsets))
    return np.vstack([trigger_onsets[:min_len], trigger_offsets[:min_len]]).T

def process_dataset(file_path, dataset_name):
    """Process a dataset and simulate real-time classification"""
    print(f"\n=== Processing {dataset_name} ===")
    
    # Load data
    print(f"Loading data from {file_path}...")
    eeg_data = load_data(file_path)
    
    if eeg_data is None:
        print(f"Failed to load data from {file_path}")
        return None
    
    # Find trigger intervals
    trigger_channel = eeg_data[9]
    trigger_intervals = find_trigger_intervals(trigger_channel)
    
    print(f"Found {len(trigger_intervals)} trigger intervals")
    
    # Expected pattern is 15Hz, 12Hz, 10Hz, 9Hz repeated 5 times
    expected_labels = np.array([1, 2, 3, 4] * 5)  # Class labels 1-4 repeated 5 times
    
    # Truncate to match actual number of intervals found
    if len(trigger_intervals) < len(expected_labels):
        print(f"Warning: Found fewer trigger intervals ({len(trigger_intervals)}) than expected ({len(expected_labels)})")
        expected_labels = expected_labels[:len(trigger_intervals)]
    elif len(trigger_intervals) > len(expected_labels):
        print(f"Warning: Found more trigger intervals ({len(trigger_intervals)}) than expected ({len(expected_labels)})")
        trigger_intervals = trigger_intervals[:len(expected_labels)]
    
    # Extract expected frequencies
    expected_freqs = [stim_freqs[label-1] for label in expected_labels]
    
    # Set up plots
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 1, 1], hspace=0.4)  # Increased spacing
    
    # Create subplots
    ax_eeg = plt.subplot(gs[0])
    ax_fbcca = plt.subplot(gs[1])
    ax_confidence = plt.subplot(gs[2])
    ax_correct = plt.subplot(gs[3])
    
    # Initialize plots with more spacing
    ax_eeg.set_title(f"SSVEP Classification: {dataset_name}", pad=10)
    ax_eeg.set_ylabel("EEG Amplitude")
    ax_eeg.set_xlabel("Time (samples)")
    
    ax_fbcca.set_title("FBCCA Correlation Values", pad=10)
    ax_fbcca.set_ylabel("Correlation")
    ax_fbcca.set_ylim(0, 1)
    
    ax_confidence.set_title("Classification Confidence", pad=10)
    ax_confidence.set_ylabel("Confidence")
    ax_confidence.set_ylim(0, 1)
    
    ax_correct.set_title("Classification Accuracy", pad=10)
    ax_correct.set_ylabel("Correct")
    ax_correct.set_ylim(-0.1, 1.1)
    ax_correct.set_yticks([0, 1])
    ax_correct.set_yticklabels(["Incorrect", "Correct"])
    
    # Process each interval
    all_results = []
    epoch_duration = 3  # seconds
    epoch_samples = int(epoch_duration * fs)
    window_step = fs // 4  # Update every 0.25 seconds
    
    for i, (start, end) in enumerate(trigger_intervals):
        # Skip if interval is too short
        if end - start < epoch_samples:
            continue
        
        # Get expected frequency
        expected_freq = expected_freqs[i]
        expected_idx = stim_freqs.index(expected_freq)
        
        # Extract EEG signals for this interval (CH2-CH9)
        eeg_signals = eeg_data[1:9, start:end]
        
        # Preprocess the signals
        preprocessed_signals = preprocess_eeg(eeg_signals, apply_notch=True, fs=fs)
        
        # Sliding window classification
        interval_results = []
        
        for j in range(0, preprocessed_signals.shape[1] - epoch_samples + 1, window_step):
            # Extract window
            window = preprocessed_signals[:, j:j+epoch_samples]
            
            # Classify
            pred_freq, confidence, corr_values = classify_epoch(window, stim_freqs, fs)
            
            # Normalize correlation values for display
            normalized_corr = np.array(corr_values) / max(max(corr_values), 1e-10)
            
            # Check if prediction is correct
            is_correct = pred_freq == expected_freq
            
            # Store results
            interval_results.append({
                'time': start + j,
                'expected_freq': expected_freq,
                'predicted_freq': pred_freq,
                'confidence': confidence,
                'corr_values': normalized_corr,
                'is_correct': is_correct
            })
        
        # Add results for this interval
        all_results.extend(interval_results)
    
    # Extract result arrays for plotting
    times = [r['time'] for r in all_results]
    expected_freqs = [r['expected_freq'] for r in all_results]
    predicted_freqs = [r['predicted_freq'] for r in all_results]
    confidences = [r['confidence'] for r in all_results]
    all_corr_values = np.array([r['corr_values'] for r in all_results])
    correct_predictions = [1 if r['is_correct'] else 0 for r in all_results]
    
    # Create plots
    ax_eeg.plot(times, predicted_freqs, 'r-', label='Predicted Frequency')
    ax_eeg.plot(times, expected_freqs, 'b--', label='Expected Frequency')
    ax_eeg.legend()
    
    # Plot FBCCA correlation values
    for i, freq in enumerate(stim_freqs):
        ax_fbcca.plot(times, all_corr_values[:, i], label=f"{freq} Hz")
    ax_fbcca.legend()
    
    # Plot confidence
    ax_confidence.plot(times, confidences)
    
    # Plot correct/incorrect
    ax_correct.plot(times, correct_predictions, 'g-')
    
    # Calculate accuracy
    accuracy = np.mean(correct_predictions)
    plt.figure(fig.number)  # Ensure we're working with the correct figure
    plt.suptitle(f"SSVEP Classification: {dataset_name}\nAccuracy: {accuracy:.2%} (Total Samples: {len(all_results)}, Window: 3s)", 
                 fontsize=16, y=0.98)  # Increased y position to prevent overlap
    
    # Save figure
    output_file = f"images/prediction_visualization_{dataset_name.replace(' ', '_').lower()}.png"
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Total samples: {len(all_results)}")
    print(f"Visualization saved to {output_file}")
    
    # Create confusion matrix
    create_confusion_matrix(predicted_freqs, expected_freqs, dataset_name)
    
    return accuracy

def create_confusion_matrix(predicted_freqs, expected_freqs, dataset_name):
    """Create a confusion matrix for the continuous predictions"""
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    # Convert frequencies to class indices (0-3)
    pred_indices = [stim_freqs.index(freq) for freq in predicted_freqs]
    expected_indices = [stim_freqs.index(freq) for freq in expected_freqs]
    
    # Calculate confusion matrix
    cm = confusion_matrix(expected_indices, pred_indices)
    
    # Calculate the total number of samples
    total_samples = len(predicted_freqs)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=freq_names, yticklabels=freq_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Calculate class distribution
    class_counts = []
    for i in range(len(stim_freqs)):
        count = np.sum(np.array(expected_indices) == i)
        class_counts.append(count)
    
    # Create a more informative title with sample counts and window info
    plt.title(f'Continuous Classification Confusion Matrix: {dataset_name}\n'
              f'Total Samples: {total_samples} (Window Size: 3s, Step: 0.25s)\n'
              f'Class Distribution: {freq_names[0]}: {class_counts[0]}, {freq_names[1]}: {class_counts[1]}, '
              f'{freq_names[2]}: {class_counts[2]}, {freq_names[3]}: {class_counts[3]}',
              pad=20)  # Add padding to prevent overlap
    
    # Save figure
    output_file = f"images/continuous_confusion_matrix_{dataset_name.replace(' ', '_').lower()}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {output_file}")
    print(f"Total samples predicted: {total_samples}")
    print(f"Window size: 3 seconds (768 samples), Step: 0.25 seconds (64 samples)")
    print(f"Class distribution: {dict(zip(freq_names, class_counts))}")

def main():
    # Dataset files and names
    datasets = [
        {"path": "data/subject_1_fvep_led_training_1.mat", "name": "Subject 1 Training 1"},
        {"path": "data/subject_1_fvep_led_training_2.mat", "name": "Subject 1 Training 2"},
        {"path": "data/subject_2_fvep_led_training_1.mat", "name": "Subject 2 Training 1"},
        {"path": "data/subject_2_fvep_led_training_2.mat", "name": "Subject 2 Training 2"}
    ]
    
    # Process each dataset
    results = {}
    for dataset in datasets:
        accuracy = process_dataset(dataset["path"], dataset["name"])
        results[dataset["name"]] = accuracy
    
    # Create summary visualization
    visualize_accuracy_results(results)

def visualize_accuracy_results(results):
    """Create a bar chart of accuracies"""
    # Filter out None values
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    # Calculate average
    average = np.mean(list(valid_results.values()))
    valid_results["Average"] = average
    
    # Create figure
    plt.figure(figsize=(10, 6))
    names = list(valid_results.keys())
    values = [v * 100 for v in valid_results.values()]
    
    bars = plt.bar(names, values)
    bars[-1].set_color('red')  # Highlight average
    
    # Add accuracy values
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{values[i]:.1f}%', ha='center', va='bottom')
    
    plt.title("Continuous Classification Accuracy")
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Save figure
    plt.tight_layout()
    plt.savefig("images/continuous_classification_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n=== Overall Continuous Classification Results ===")
    for name, accuracy in valid_results.items():
        print(f"{name}: {accuracy:.1%}")

if __name__ == "__main__":
    main() 