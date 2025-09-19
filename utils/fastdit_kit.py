import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, features_dir):
        # features_dir, _features/_labels
        L = os.listdir(features_dir)
        print(f'---> Folders in {features_dir}: {L}')
        for name in L:
            if name.endswith('_features'):
                self.features_dir = os.path.join(features_dir, name)
            elif name.endswith('_labels'):
                self.labels_dir = os.path.join(features_dir, name)

        self.features_files = sorted(os.listdir(self.features_dir))
        self.labels_files = sorted(os.listdir(self.labels_dir))
        assert len(self.features_files) == len(self.labels_files) == 1281167 # ImageNet


    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        aug_id = torch.randint(0,2,size=(1,)).item()
        features = features[aug_id]
        return torch.from_numpy(features), torch.from_numpy(labels).squeeze(0)