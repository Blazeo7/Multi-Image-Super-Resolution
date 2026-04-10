import torch
import numpy as np
import cv2


class LocalDescriptor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.batch_size = 64
        self.crop_size = 64

    def filter_points(self, image: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Filter points that are out of the image bounds."""
        mask = np.logical_and.reduce(
            [
                points[:, 0] >= self.crop_size / 2 + 1,
                points[:, 0] < image.shape[1] - self.crop_size / 2 - 1,
                points[:, 1] >= self.crop_size / 2 + 1,
                points[:, 1] < image.shape[0] - self.crop_size / 2 - 1,
            ]
        )
        return points[mask], mask

    def process_page(self, image: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Crop patches from the image and compute descriptors for each patch."""
        patches = []
        for point in points:
            x, y = point
            x = int(x) - self.crop_size // 2
            y = int(y) - self.crop_size // 2
            patch = image[y : y + self.crop_size, x : x + self.crop_size]
            patches.append(patch)

        descriptors = self(patches)
        return descriptors

    def filter_points_kp(
        self, image: np.ndarray, points: list[cv2.KeyPoint]
    ) -> list[cv2.KeyPoint]:
        """Filter keypoints that are out of the image bounds."""
        mask = np.logical_and.reduce(
            [
                [point.pt[0] >= self.crop_size / 2 + 1 for point in points],
                [
                    point.pt[0] < image.shape[1] - self.crop_size / 2 - 1
                    for point in points
                ],
                [point.pt[1] >= self.crop_size / 2 + 1 for point in points],
                [
                    point.pt[1] < image.shape[0] - self.crop_size / 2 - 1
                    for point in points
                ],
            ]
        )
        return [point for point, m in zip(points, mask) if m]

    def process_page_kp(
        self, image: np.ndarray, points: list[cv2.KeyPoint]
    ) -> torch.Tensor:
        """Crop patches from the image and compute descriptors for each patch."""
        patches = []
        for point in points:
            x, y = point.pt
            x = int(x) - self.crop_size // 2
            y = int(y) - self.crop_size // 2
            patch = image[y : y + self.crop_size, x : x + self.crop_size]
            patches.append(patch)

        descriptors = self(patches)
        return descriptors

    def __call__(self, images: torch.Tensor | list[np.ndarray]) -> torch.Tensor:
        results = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = images[i : i + self.batch_size]
                if type(batch) == list:
                    batch = np.stack(batch, axis=0)
                    batch = torch.from_numpy(batch)
                batch = batch.permute(0, 3, 1, 2)

                assert batch.dtype == torch.uint8
                batch = batch.to(self.device)
                batch_embedding = self.model(batch)
                results.append(batch_embedding)

            results = torch.cat(results, dim=0)
            return results
