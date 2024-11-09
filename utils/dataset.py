import os
import cv2
import numpy as np

import torch
import torch.utils.data as data


class WiderFaceDetection(data.Dataset):
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = []
        self.words = []
        self._parse_labels(self.root)

    def _parse_labels(self, root):
        with open(os.path.join(root, 'label.txt'), 'r') as f:
            lines = f.read().splitlines()

        labels = []
        for line in lines:
            if line.startswith('#'):
                if labels:
                    self.words.append(labels.copy())
                    labels.clear()
                image_path = os.path.join(root, 'images', line[2:])
                self.image_paths.append(image_path)
            else:
                labels.append([float(x) for x in line.split(' ')])
        self.words.append(labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        height, width, _ = image.shape

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if not labels:
            return torch.from_numpy(image), annotations

        for label in labels:
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            annotation[0, 14] = 1 if label[4] >= 0 else -1

            annotations = np.append(annotations, annotation, axis=0)

        target = np.array(annotations)
        if self.transform is not None:
            image, target = self.transform(image, target)

        return torch.from_numpy(image), target

    @staticmethod
    def collate_fn(batch):
        images = []
        targets = []
        # Iterate over each data sample in the batch
        for image, target in batch:
            images.append(image)
            targets.append(torch.from_numpy(target).float())

        return torch.stack(images, 0), targets
