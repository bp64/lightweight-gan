import pytest
from lightweight_gan.dataset import ImageDataset


@pytest.fixture
def jpeg_image():
    jpg = BytesIO()
    im = Image.new("RGB", (100, 200), color=200)
    im.save(jpg, format="jpeg")
    yield jpg
    im.close()


@pytest.fixture
def png_image():
    png = BytesIO()
    im = Image.new("RGB", (100, 200), color=200)
    im.save(png, format="png")
    yield png
    png.close()


@pytest.fixture
def create_png_data_dir(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("png")
    yield data_dir.as_posix()


@pytest.fixture
def png_jpg_images(png_image, jpeg_image):
    yield png_image, jpeg_image


def test_dataset_opens_jpeg(jpeg_image):
    ImageDataset()
    assert True
