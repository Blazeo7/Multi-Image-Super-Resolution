import json

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class MISRDataset(Dataset):
    def __init__(self, manifest_path, transform=None):
        """
        Expects a manifest file in JSON format with the following structure:
            {
                "scale_factor": 2,
                "num_aligned_images": 4,
                "samples": [
                    {
                        "id": "sample_id_001",
                        "reference": "path/to/lr/reference.png",
                        "supporting": ["path/to/lr/s1.png", "path/to/lr/s2.png", ...],
                        "target": "path/to/hr/target.png"
                    },
                    ...
                ]
            }
        """
        self.transform = transform
        self.manifest = self._load_manifest(manifest_path)
        self.samples = self.manifest["samples"]
        self.scale_factor = self.manifest["scale_factor"]

    def __len__(self):
        return len(self.samples)

    def _load_manifest(self, manifest_path):
        assert manifest_path.endswith(".json"), "Manifest must be a JSON file"
        with open(manifest_path, "r") as f:
            return json.load(f)

    def _load_image(self, path):
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return to_tensor(image)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        reference = self._load_image(sample["reference"])
        supporting = [self._load_image(p) for p in sample["supporting"]]
        target = to_tensor(Image.open(sample["target"]).convert("RGB"))

        frames = torch.cat([reference] + supporting, dim=0)  # (num_frames * C, H, W) e.g. (15, H, W)
        return frames, target
