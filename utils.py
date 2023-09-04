import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from typing import Tuple, Optional
from model import Patches


class DataProcessing:
    """
    Attributes:
        batch_size (int): Size of the batches of data.
        image_size (Tuple[int, int]): Size of image after reading, specified as (height, width).
        shuffle (bool): Whether to shuffle the data.
        seed (int): Optional random seed for shuffling and transformations.
    """
    def __init__(
        self,
        batch_size: int = 32,
        image_size: Tuple[int, int] = (256, 256),
        shuffle: bool = True,
        seed: int = 42,
        as_supervised: bool = True,
        with_info: bool = True,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.seed = seed
        self.with_info = with_info
        self.as_supervised = as_supervised

    def CreateData(self, *dataDirList) -> tuple:
        """
        :param dataDirList: List of directory paths with data.
        :return: List of image datasets.
        """
        result = list()
        for dataDir in dataDirList:
            data = tf.keras.utils.image_dataset_from_directory(
                directory=dataDir,
                batch_size=self.batch_size,
                image_size=self.image_size,
                shuffle=self.shuffle,
                seed=self.seed,
            )
            result.append(data)

        return tuple(result)

    def LoadData(self, name: str, split: list) -> tuple:

        ds, info = tfds.load(
                name=name,
                split=split,
                as_supervised=self.as_supervised,
                batch_size=self.batch_size,
                shuffle_files=self.shuffle,
                with_info=self.with_info,
        )
        return ds, info


def plot_image(
    data: Optional, class_names: dict, size: Tuple[int, int] = (8, 8)
) -> None:
    """
    :param data: Image dataset.
    :param class_names: Class names of data.
    :param size: Figure size.
    :return: None
    """

    plt.figure(figsize=size)

    iterator = iter(data)
    images, labels = next(iterator)

    plt.imshow(images[0].numpy().astype("uint8"))
    plt.title(class_names[str(labels[0].numpy())])
    plt.axis('off')
    plt.show()


def plot_patches(
    data: Optional, patch_size: int, size: Tuple[int, int] = (6, 6),
) -> None:
    """

    :param data: Image dataset.
    :param patch_size: Image patch size.
    :param size: Figure size.
    :return: None
    """

    iterator = iter(data)
    images, _ = next(iterator)

    image_size = images.shape[1]
    resized_image = tf.image.resize(tf.convert_to_tensor([images[0]]), size=(image_size, image_size))
    patches = Patches(patch_size)(resized_image)

    n = int(np.sqrt(patches.shape[1]))

    plt.figure(figsize=size)

    for i, patch in enumerate(patches[0]):
        plt.subplot(n, n, i + 1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
        plt.imshow(patch_img.numpy().astype("uint8"))
        plt.axis("off")

    plt.show()
