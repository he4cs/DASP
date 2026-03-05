import os
import json
import csv
from typing import Sequence

from datasets.AudiosetDataset import AudiosetDataset


# Support for VGGSound and Kinetics-50
def create_audiosetc_dataset(
    dataset_name: str = 'vggsound',
    data_dir: str = './data/VGGSound',
    corruption: dict = {'video': 'gaussian_noise', 'audio': 'gaussian_noise'},
    corruption_seq: Sequence[dict] = None,
    severity: int = 5,
    setting: str = 'continual'):
    
    label_csv = os.path.join(f'datasets/{dataset_name}_list', 'class_labels_indices.csv')
    test_json = os.path.join(f'datasets/{dataset_name}_list', 'test_list.json')
    
    with open(test_json, 'r') as f:
        clean_samples = json.load(f)
        clean_samples = clean_samples['data']
    with open(label_csv, 'r') as f:
        class_to_idx = csv.DictReader(f)
        class_to_idx = {row['mid']: int(row['index']) for row in class_to_idx}
    
    corruption_seq = corruption_seq if "mixed_domains" in setting else [corruption]
    
    corrupted_samples = []
    for corruption in corruption_seq:
        for sample in clean_samples:
            
            video_folder = sample['video_path'].split('/')[0]
            video_path = os.path.join(data_dir, video_folder + '-C', corruption['video'], f'severity_{severity}') if 'video' in corruption else os.path.join(data_dir, sample['video_path'])
            
            audio_folder, filename = sample['wav'].split('/')
            audio_path = os.path.join(data_dir, audio_folder + '-C', corruption['audio'], f'severity_{severity}', filename) if 'audio' in corruption else os.path.join(data_dir, sample['wav'])
            
            corrupted_samples.append({
                'video_id': sample['video_id'],
                'video_path': video_path,
                'wav': audio_path,
                'labels': class_to_idx[sample['labels']],
            })
    
    return AudiosetDataset(samples=corrupted_samples, mode='eval')


# Support for CMU-MOSI
def create_mosic_dataset():
    pass