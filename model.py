import tensorflow as tf
from keras import layers


class DataAugmentation(layers.Layer):
    """
    Attributes:
        image_size (int): Output image height and width size.
        mode (str): String indicating which flip mode to use.
        factor (float): Rotation value of the image.
        height_factor (float): A tuple of size 2 representing lower and upper bound for zooming vertically.
        width_factor (float): A tuple of size 2 representing lower and upper bound for zooming horizontally.
    """
    def __init__(self, image_size: int, mode: str,  factor: float, height_factor: float, width_factor: float):
        super().__init__()
        self.image_size = image_size
        self.mode = mode
        self.factor = factor
        self.height_factor = height_factor
        self.width_factor = width_factor

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:

        x = layers.Normalization()(inputs)
        x = layers.Resizing(height=self.image_size, width=self.image_size)(x)
        x = layers.RandomFlip(mode=self.mode)(x)
        x = layers.RandomRotation(factor=self.factor)(x)
        x = layers.RandomZoom(height_factor=self.height_factor,
                              width_factor=self.width_factor)(x)
        return x


class MLP(layers.Layer):
    """
    Attributes:
        hidden_units (list): List of dimensions of output space.
        dropout_rate (float): Fraction of the input units to drop.
    """
    def __init__(self, hidden_units: list, dropout_rate: float):
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        for units in self.hidden_units:
            x = layers.Dense(units=units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(rate=self.dropout_rate)(x)
        return x


class Patches(layers.Layer):
    """
    Attributes:
        patch_size (int): Image patch size.
    """
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def __call__(self, images: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    """
    Attributes:
        num_patches (int): Image patch size.
        projection (layers.Dense): Linear transform layer.
        position_embedding (layers.Dense): Learnable position embedding to the projected vector.
    """
    def __init__(self, patch_size: int, image_size: int, projection_dim: int):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches, output_dim=projection_dim
        )

    def __call__(self, patch: tf.Tensor) -> tf.Tensor:
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
