from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


DISEASE_COLUMNS = [
    "diabetic_retinopathy",
    "macular_edema",
    "scar",
    "nevus",
    "amd",
    "vascular_occlusion",
    "hypertensive_retinopathy",
    "drusens",
    "hemorrhage",
    "retinal_detachment",
    "myopic_fundus",
    "increased_cup_disc",
    "other",
]

CATEGORY_TO_INDEX = {
    "no_disease": 0,
    "single_disease": 1,
    "multiple_disease": 2,
}


# def default_image_transform(image_size: int | tuple[int, int] = 224) -> transforms.Compose:
#     if isinstance(image_size, int):
#         image_size = (image_size, image_size)

#     return transforms.Compose(
#         [
#             transforms.Resize(image_size),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=(0.485, 0.456, 0.406),
#                 std=(0.229, 0.224, 0.225),
#             ),
#         ]
#     )


class BRSETImageDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path,
        split: str | None = None,
        disease_categories: Iterable[str] | None = None,
        transform=None,
        label_columns: list[str] | None = None,
        image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
        validate_files: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.label_columns = label_columns or DISEASE_COLUMNS
        self.image_extensions = image_extensions

        df = pd.read_csv(self.csv_path)

        if split is not None:
            df = df[df["split"].eq(split)].copy()

        if disease_categories is not None:
            categories = set(disease_categories)
            df = df[df["disease_category"].isin(categories)].copy()

        self.df = df.reset_index(drop=True)

        missing_columns = [
            column
            for column in ["image_id", "patient_id", "disease_count", "disease_category"]
            + self.label_columns
            if column not in self.df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns in {self.csv_path}: {missing_columns}")

        if validate_files:
            missing = [
                image_id
                for image_id in self.df["image_id"]
                if self._resolve_image_path(str(image_id)) is None
            ]
            if missing:
                preview = ", ".join(map(str, missing[:10]))
                raise FileNotFoundError(
                    f"Could not find {len(missing)} images in {self.image_dir}. "
                    f"First missing image_ids: {preview}"
                )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        image_id = str(row["image_id"])
        image_path = self._resolve_image_path(image_id)

        if image_path is None:
            raise FileNotFoundError(
                f"Image for image_id={image_id} was not found under {self.image_dir}"
            )

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        labels = torch.tensor(row[self.label_columns].astype("float32").to_numpy())
        disease_category = str(row["disease_category"])

        return {
            "image": image,
            "labels": labels,
            "disease_count": torch.tensor(int(row["disease_count"]), dtype=torch.long),
            "disease_category": torch.tensor(
                CATEGORY_TO_INDEX[disease_category], dtype=torch.long
            ),
            "disease_category_name": disease_category,
            "image_id": image_id,
            "patient_id": int(row["patient_id"]),
            "image_path": str(image_path),
        }

    def _resolve_image_path(self, image_id: str) -> Path | None:
        raw_path = self.image_dir / image_id
        if raw_path.exists():
            return raw_path

        if Path(image_id).suffix:
            return None

        for extension in self.image_extensions:
            candidate = self.image_dir / f"{image_id}{extension}"
            if candidate.exists():
                return candidate

        return None


def create_brset_dataloader(
    csv_path: str | Path,
    image_dir: str | Path,
    split: str,
    batch_size: int = 32,
    image_size: int | tuple[int, int] = 224,
    shuffle: bool | None = None,
    num_workers: int = 4,
    disease_categories: Iterable[str] | None = None,
    transform=None,
    validate_files: bool = False,
) -> DataLoader:
    if transform is None:
        transform = default_image_transform(image_size)

    if shuffle is None:
        shuffle = split == "retrieval"

    dataset = BRSETImageDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        split=split,
        disease_categories=disease_categories,
        transform=transform,
        validate_files=validate_files,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    retrieval_loader = create_brset_dataloader(
        csv_path="brset_dataset_distribution.csv",
        image_dir="/path/to/brset/images",
        split="retrieval",
        batch_size=16,
    )
    test_loader = create_brset_dataloader(
        csv_path="brset_dataset_distribution.csv",
        image_dir="/path/to/brset/images",
        split="test",
        batch_size=16,
        shuffle=False,
    )

    print(f"retrieval images: {len(retrieval_loader.dataset)}")
    print(f"test images: {len(test_loader.dataset)}")
