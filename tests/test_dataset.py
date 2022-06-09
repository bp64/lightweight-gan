import os
from io import BytesIO
from itertools import product
from pathlib import Path

import pytest
from lightweight_gan.dataset import ImageDataset
from PIL import Image
from pytest import FixtureRequest, TempPathFactory

FORMATS = ["PNG", "JPEG"]
MODES = ["RGB", "RGBA"]
SIZES = [(128, 128), (128, 256), (256, 128)]


def create_image(mode: str, size: tuple[int], format: str) -> BytesIO:
    buf = BytesIO()
    with Image.new(mode=mode, size=size) as im:
        im.save(buf, format=format)
    return buf


@pytest.fixture(scope="session", params=FORMATS)
def img_dir(tmp_path_factory: TempPathFactory, request: FixtureRequest) -> Path:
    format = request.param
    img_dir = tmp_path_factory.mktemp(format, numbered=False)
    img_configs = product(MODES, SIZES)
    for mode, size in img_configs:
        bytes = create_image(mode=mode, size=size, format=format)
        with open(img_dir / f"{mode}_{size}.png", mode="wb") as f:
            f.write(bytes.getbuffer())
    yield img_dir


@pytest.fixture(scope="session")
def images_dir(tmp_path_factory: TempPathFactory) -> Path:
    image_dir = tmp_path_factory.mktemp("images")
    yield image_dir


def test_dataset_opens_png(img_dir: Path):
    dataset = ImageDataset(folder=img_dir, image_size=100)
    assert True
