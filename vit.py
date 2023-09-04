import json
from keras import layers, Model
from model import DataAugmentation, Patches, PatchEncoder, MLP


def get_hyperparameters(path):
    """
    :param path: Path to json file with Hyperparameters.
    :return: Hyperparameters dictionary.
    """
    with open(path, "r") as file:
        hyperparameters = json.load(file)
    return hyperparameters


def create_vit_classifier(hp, patch_size, transformer_layers, num_heads):
    """
    :param hp: Hyperparameters dictionary.
    :param patch_size: Image patch size.
    :param transformer_layers: Number of transfer layers.
    :param num_heads: Size number of attention heads.
    :return: Vit model.
    """

    inputs = layers.Input(shape=tuple(hp["input_shape"]))

    augmentation = DataAugmentation(
                   image_size=hp["image_size"],
                   mode=hp["mode"],
                   factor=hp["factor"],
                   height_factor=hp["height_factor"],
                   width_factor=hp["width_factor"],
                   )

    augmented = augmentation(inputs)

    patches = Patches(patch_size)(augmented)

    encoded_patches = PatchEncoder(patch_size, hp["image_size"], hp["projection_dim"])(patches)

    for _ in range(transformer_layers):

        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=hp["projection_dim"], dropout=0.1
        )(x1, x1)

        x2 = layers.Add()([attention_output, encoded_patches])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        x3 = MLP(hidden_units=hp["transformer_units"], dropout_rate=0.1)(x3)

        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    features = MLP(hidden_units=hp["mlp_head_units"], dropout_rate=0.5)(representation)

    logits = layers.Dense(hp["num_classes"])(features)

    return Model(inputs=inputs, outputs=logits)
