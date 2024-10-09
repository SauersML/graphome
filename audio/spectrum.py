import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import librosa
import soundfile as sf
import os
import time

def z_normalize(data):
    """Z-normalize the data (zero mean, unit variance)."""
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std if std != 0 else data

def generate_audio_from_csv(csv_path, columns_per_sec, output_wav_path, boost_factor=10, gain_increase_db=80):
    """Generate audio from CSV spectrogram data using librosa and save with soundfile."""
    print("Loading CSV for audio creation...")
    data = pd.read_csv(csv_path, header=None).to_numpy()

    print("Z-normalizing data for audio...")
    data = z_normalize(data)

    # Reshape the CSV data into a 2D array that represents the spectrogram
    spectrogram = data.T

    # Define hop length to match the number of columns per second for consistency
    hop_length = int(spectrogram.shape[1] / columns_per_sec)

    # Convert spectrogram to audio using Short-Time Fourier Transform (ISTFT)
    print("Converting spectrogram to audio using librosa...")
    audio_signal = librosa.istft(spectrogram, hop_length=hop_length)

    # Apply gain increase
    print(f"Increasing gain by {gain_increase_db} dB...")
    gain_linear = 10 ** (gain_increase_db / 20)
    audio_signal *= gain_linear

    # Normalize volume
    print("Normalizing audio volume...")
    audio_signal /= np.max(np.abs(audio_signal))

    # Save audio as WAV file
    print(f"Saving audio to WAV file: {output_wav_path}...")
    sf.write(output_wav_path, audio_signal, int(columns_per_sec * hop_length / spectrogram.shape[0]))  # Ensure the correct sample rate
    print("Audio creation complete.")

def create_heatmap_video(csv_path, columns_per_sec, output_video_path, output_wav_path, boost_factor=10, gain_increase_db=80):
    """Create a video from CSV spectrogram data with audio and moving bar."""
    print("Loading CSV for heatmap video creation...")
    data = pd.read_csv(csv_path, header=None).to_numpy()

    print("Z-normalizing data for heatmap (no Fourier transform applied)...")
    normalized_data = z_normalize(data)

    cmap = plt.get_cmap('coolwarm')
    cmap.set_bad(color='black')

    # Mask data below a certain threshold (e.g., very low values) to create "missing" regions
    masked_data = np.ma.masked_where(np.abs(normalized_data) < 1e-9, normalized_data)

    # Create heatmap plot (use original z-normalized data for visualizing)
    print("Creating heatmap on black background inside the plot...")
    fig, ax = plt.subplots()
    cax = ax.imshow(masked_data, cmap=cmap, aspect='auto', origin='lower', interpolation='none', vmin=-2, vmax=2)

    fig.patch.set_facecolor('white')
    ax.set_facecolor('black')

    ax.set_xlabel('Time')
    ax.set_ylabel('')
    fig.colorbar(cax)
    ax.set_title("Z-Normed Spectrogram Heatmap")

    # Progress tracking for video frames
    frames = normalized_data.shape[1]
    print(f"Generating video with {frames} frames...")

    def update(i):
        progress = (i / frames) * 100
        if i % (frames // 10) == 0:  # Print every 10% progress
            print(f"Video creation progress: {progress:.2f}%")
        ax.clear()
        ax.imshow(masked_data, cmap=cmap, aspect='auto', origin='lower', interpolation='none', vmin=-2, vmax=2)
        ax.axvline(x=i, color='cyan', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        ax.set_title(f"Time: {i / columns_per_sec:.2f} seconds")

    # Save the video
    print(f"Saving video to: {output_video_path}...")
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000/columns_per_sec)
    ani.save(output_video_path, writer='ffmpeg', fps=columns_per_sec)
    print("Video saved.")

    # Generate the audio file
    print("Generating audio for video...")
    generate_audio_from_csv(csv_path, columns_per_sec, output_wav_path, boost_factor, gain_increase_db)
    print("Heatmap video creation complete.")

# Example run
csv_path = '/Users/scott/Downloads/audio/submatrix.eigenvectors.csv'
columns_per_sec = 50
output_video_path = 'output_heatmap_video.mp4'
output_wav_path = 'output_audio.wav'

print("Starting heatmap video and audio creation process...")
create_heatmap_video(csv_path, columns_per_sec, output_video_path, output_wav_path)
print("Process complete.")
