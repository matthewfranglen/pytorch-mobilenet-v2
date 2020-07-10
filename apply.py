#!/usr/bin/env python

"""
Apply the network to an image
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

import PIL  # type: ignore
import torch
import torch.hub
import torch.utils.data
import typer
from torchvision import datasets, transforms  # type: ignore


class SingleImageDataset(datasets.VisionDataset):
    def __init__(
        self,
        image: Path,
        transform: Optional[Callable[[PIL.Image.Image], torch.Tensor]] = None,
    ):
        super().__init__(
            image,
            transform=transform if transform else transforms.ToTensor(),
            target_transform=None,
        )
        self.image = PIL.Image.open(image)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.image

        if self.transform is not None:
            img = self.transform(img)

        return img, 0


def main(
    file: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True
    ),
    cuda: bool = True,
):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    dataset = SingleImageDataset(
        file, transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    dataloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset, batch_size=1
    )
    model = torch.hub.load(
        "matthewfranglen/pytorch-mobilenet-v2", "mobilenet_v2", pretrained=True
    ).eval()

    device = "cuda" if torch.cuda.is_available() and cuda else "cpu"
    model = model.to(device)
    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.to(device)
            output = model(batch)
            print(output.cpu().argmax().item())


if __name__ == "__main__":
    typer.run(main)
