import csv
import json
import logging
import os.path
import numpy as np
import random
from PIL import Image
import PIL

import torch
import torch.nn.functional
from torch.utils.data import Dataset
import torchaudio
import torchvision.transforms as T

logger = logging.getLogger(__name__)


class AudiosetDataset(Dataset):
    def __init__(self, samples, mode='eval',
                 melbins=128, target_length=1024, norm_mean=-5.081, norm_std=4.4849, # audio config
                 frame_use=-1, total_frame=10, im_res=224 # video config
                 ):
        """
        Dataset that manages audio recordings
        """
        self.mode = mode # train or eval
        self.samples = samples
        
        self.melbins = melbins
        self.target_length = target_length
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        
        # set the frame to use in the eval mode, default value for training is -1 which means random frame
        self.frame_use = frame_use
        # by default, 10 frames are used
        self.total_frame = total_frame
        # by default, all models use 224*224, other resolutions are not tested
        self.im_res = im_res

        self.preprocess = T.Compose([
            T.Resize(self.im_res, interpolation=PIL.Image.BICUBIC),
            T.CenterCrop(self.im_res),
            T.ToTensor(),
            T.Normalize(
                mean=[0.4850, 0.4560, 0.4060],
                std=[0.2290, 0.2240, 0.2250]
            )])

    def _wav2fbank(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        
        target_length = self.target_length
        n_frames = fbank.shape[0]
        p = target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0: target_length, :]
        
        fbank = (fbank - self.norm_mean) / (self.norm_std)
        return fbank

    def rand_select_image(self, video_id, video_path):
        if self.mode == 'eval':
            # if not specified, use the middle frame
            if self.frame_use == -1:
                frame_idx = int((self.total_frame) / 2)
            else:
                frame_idx = self.frame_use
        else:
            frame_idx = random.randint(0, 9)
        
        while os.path.exists(os.path.join(video_path, f'frame_{frame_idx}', f'{video_id}.jpg')) == False and frame_idx >= 1:
            print(f'frame {frame_idx} of video {video_id} does not exist')
            frame_idx -= 1
        img_path = os.path.join(video_path, f'frame_{frame_idx}', f'{video_id}.jpg')
        return img_path
    
    def load_image(self, img_path):
        img = Image.open(img_path)
        image_tensor = self.preprocess(img)
        return image_tensor

    def __getitem__(self, index):
        sample = self.samples[index]

        try:
            fbank = self._wav2fbank(sample['wav'])
        except Exception as e:
            logger.warning(f'Error loading audio {sample["wav"]}: {e}')
            fbank = torch.zeros([self.target_length, 128]) + 0.01
        
        try:
            image = self.load_image(self.rand_select_image(sample['video_id'], sample['video_path']))
        except Exception as e:
            logger.warning(f'Error loading image from {sample["video_path"]}: {e}')
            image = torch.zeros([3, self.im_res, self.im_res]) + 0.01
        
        label = sample['labels']
        
        # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank, image, label

    def __len__(self):
        return len(self.samples)