import tensorflow as tf
import keras_tuner as kt
from vit import create_vit_classifier
from vit import get_hyperparameters


class HyperModel(kt.HyperModel):

    hyperparameters = get_hyperparameters("parameters.json")

    def build(self, hp):

        patch_size = hp.Choice("patch_size", [8, 16, 32])

        transformer_layers = hp.Int("transformer_layers", min_value=4, max_value=12, step=2)

        num_heads = hp.Choice("num_heads", [4, 8, 16])

        learning_rate = hp.Choice("learning_rate", [0.001])

        weight_decay = hp.Choice("weight_decay", [0.0001])

        model = create_vit_classifier(
            self.hyperparameters,
            patch_size,
            transformer_layers,
            num_heads,
        )

        optimizer = tf.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            ],
        )

        return model

    def __fit__(self, dataset, val):

        return self.model.fit(
            dataset=dataset,
            batch_size=self.hyperparameters["batch_size"],
            epochs=self.hyperparameters["epochs"],
            validation_data=val,
        )
