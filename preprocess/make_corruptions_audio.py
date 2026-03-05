import os
import argparse
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import concurrent.futures
from tqdm import tqdm
import warnings

warnings.simplefilter("ignore", UserWarning)

# ==========================================
# Data Loader Utils
# ==========================================

AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a')

def is_audio_file(filename):
    """Checks if a file is an audio file based on its extension."""
    return filename.lower().endswith(AUDIO_EXTENSIONS)

# Lazy-loading cache for background noises to prevent massive I/O bottlenecks
_WEATHER_AUDIO_CACHE = {}

def get_weather_audio(path):
    """Loads weather audio once per process and serves from memory thereafter."""
    global _WEATHER_AUDIO_CACHE
    if path not in _WEATHER_AUDIO_CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing external noise file: {path}")
        _WEATHER_AUDIO_CACHE[path] = AudioSegment.from_file(path)
    return _WEATHER_AUDIO_CACHE[path]

# ==========================================
# Distortions
# ==========================================

def apply_gaussian_noise(audio_path, output_path, severity):
    audio, sr = sf.read(audio_path)
    noise_std = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    
    noise = np.random.normal(0, noise_std, len(audio))
    audio_with_noise = audio + noise

    sf.write(output_path, audio_with_noise, sr)

def apply_external_noise(audio_path, weather_path, output_path, severity):
    audio = AudioSegment.from_file(audio_path)
    background_sound = get_weather_audio(weather_path)

    # Adjust the length to match the source audio
    if len(audio) <= len(background_sound):
        background_sound = background_sound[:len(audio)]
    else:
        num_repeats = (len(audio) // len(background_sound)) + 1
        background_sound = background_sound * num_repeats
        background_sound = background_sound[:len(audio)]

    scale = [1, 2, 4, 6, 8]
    background_sound = background_sound.apply_gain(scale[severity - 1])

    output = audio.overlay(background_sound)
    output.export(output_path, format="wav")

# ==========================================
# Multi-processing Orchestration
# ==========================================

def process_single_audio(audio_path, output_path, corruption, severity, weather_path):
    """Worker function to process a single audio file."""
    try:
        if corruption == 'gaussian_noise':
            apply_gaussian_noise(audio_path, output_path, severity)
        else:
            weather_file = os.path.join(weather_path, f"{corruption}.wav")
            apply_external_noise(audio_path, weather_file, output_path, severity)
    except Exception as e:
        print(f"Failed to process {os.path.basename(audio_path)}: {e}")

def save_distorted_data(corruption, severity, data_path, save_path, weather_path):
    """Prepares tasks and dispatches them to a process pool."""
    
    # Define and create target directory upfront
    target_dir = os.path.join(save_path, corruption, f'severity_{severity}')
    os.makedirs(target_dir, exist_ok=True)
    
    tasks = []
    
    # Gather all valid audio files
    if not os.path.exists(data_path):
        print(f"Source directory not found: {data_path}")
        return

    for name in sorted(os.listdir(data_path)):
        if is_audio_file(name):
            audio_path = os.path.join(data_path, name)
            output_path = os.path.join(target_dir, name)
            
            # Change extension to .wav if using pydub export
            if corruption != 'gaussian_noise':
                base_name = os.path.splitext(name)[0]
                output_path = os.path.join(target_dir, f"{base_name}.wav")
                
            tasks.append((audio_path, output_path, corruption, severity, weather_path))

    if not tasks:
        print(f"No valid audio files found in {data_path}")
        return

    # Execute tasks concurrently
    workers = min(os.cpu_count() or 4, 32)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_audio, *task) for task in tasks]
        
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc=f"{corruption} (Severity {severity})"):
            pass

# ==========================================
# CLI Entry Point
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--corruption', type=str, default='gaussian_noise', 
                        choices=['all', 'gaussian_noise', 'traffic', 'crowd', 'rain', 'thunder', 'wind'])
    parser.add_argument('--severity', type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--data_path', type=str, default='data/VGGSound/audio_test')
    parser.add_argument('--save_path', type=str, default='data/VGGSound/audio_test-C')
    parser.add_argument('--weather_path', type=str, default='preprocess/weather_audios/')
    args = parser.parse_args()

    if args.corruption == 'all':
        corruption_list = ['gaussian_noise', 'traffic', 'crowd', 'rain', 'thunder', 'wind']
    else:
        corruption_list = [args.corruption]

    for corruption in corruption_list:
        save_distorted_data(
            corruption=corruption,
            severity=args.severity,
            data_path=args.data_path,
            save_path=args.save_path,
            weather_path=args.weather_path
        )