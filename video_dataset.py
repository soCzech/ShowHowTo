import os
import glob
import json
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ShowHowToDataset(Dataset):
    def __init__(self, root_path, video_length=None):
        self.root_path = root_path
        self.video_length = video_length

        with open(os.path.join(root_path, "prompts.json"), "r") as f:
            self.prompts = json.load(f)

        data = sorted(glob.glob(os.path.join(root_path, "imgseqs*", "*.jpg")))
        self.data = [x for x in data if os.path.basename(x).replace(".jpg", "") in self.prompts]
        if self.video_length is not None:
            self.data = [x for x in self.data if len(self.prompts[os.path.basename(x).replace(".jpg", "")]) >= self.video_length]
            print(f"Found {len(self.data)} images with valid prompts and length >= {video_length}, other {len(data) - len(self.data)} removed")
        else:
            print(f"Found {len(self.data)} images with valid prompts, other {len(data) - len(self.data)} removed")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_fn = self.data[idx]
        vid_id = os.path.basename(img_fn).replace(".jpg", "")
        assert vid_id in self.prompts, f"prompt file not found for {img_fn} in prompts file!"
        prompts = self.prompts[vid_id]

        img = np.array(Image.open(img_fn))
        h, w = img.shape[:2]
        w = w // len(prompts)

        imgs = [img[:, i * w:(i + 1) * w] for i in range(len(prompts))]
        imgs = np.stack(imgs, axis=0)
        if w < h:
            print(f"{img_fn} has width {w} and height {h}! Skipping...", flush=True)
            return self.__getitem__(random.randint(0, len(self.data) - 1))
        else:
            imgs = imgs[:, :, (w - h) // 2:][:, :, :h]

        indices = np.arange(len(prompts))
        if self.video_length is not None:
            indices = indices[np.random.randint(0, len(prompts) - self.video_length + 1):][:self.video_length]

        selected_prompts = [prompts[i] for i in indices]
        selected_imgs = imgs[indices]

        selected_imgs = np.stack([np.array(Image.fromarray(fr).resize((256, 256))) for fr in selected_imgs], axis=0)
        video_frames = torch.from_numpy(selected_imgs.copy()).float().div_(255 / 2).sub_(1).permute(3, 0, 1, 2)

        return video_frames, selected_prompts

    def __repr__(self):
        string = f"ShowHowToDataset(n_samples: {self.__len__()})"
        return string


class RepeatedDataset(Dataset):

    def __init__(self, ds, epoch_len):
        self.ds = ds
        self.epoch_len = epoch_len

    def __getitem__(self, idx):
        return self.ds[random.randint(0, len(self.ds) - 1)]

    def __len__(self):
        return self.epoch_len

    def __repr__(self):
        string = f"RepeatedDataset(ds: {self.ds}, epoch_len: {self.epoch_len})"
        return string


def sequence_collate(batch):
    video_frames, prompt = zip(*batch)
    video_frames = torch.stack(video_frames)
    if isinstance(prompt[0], list):
        prompt = []
        for i in range(len(batch)):
            prompt.extend(batch[i][1])
    return video_frames, prompt
