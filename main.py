import keras_tuner as kt
from utils import DataProcessing
from train import HyperModel
from utils import plot_image, plot_patches


if __name__ == "__main__":

    train = "dataset/train"
    test = "dataset/test"
    val = "dataset/val"
    class_names = {"0": "angular_leaf_spot", "1": "bean_rust", "2": "healthy"}

    dp = DataProcessing(
        batch_size=32,
        image_size=(256, 256),
        shuffle=True,
        seed=42,
    )

    train, test, val = dp.CreateData(train, test, val)

    plot_image(data=train, class_names=class_names, size=(5, 5))

    plot_patches(data=val, patch_size=16, size=(8, 8))

    tuner = kt.RandomSearch(HyperModel(),
                            objective='val_loss',
                            overwrite=True,
                            directory="tuner_dir",
                            project_name="tune_hypermodel",
                            max_trials=1,
                            max_consecutive_failed_trials=1,)

    tuner.search(train, epochs=1, validation_data=val)
