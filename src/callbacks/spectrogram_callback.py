# Adapted from:
# https://github.com/adobe-research/DeepAFx-ST

import numpy as np
import pytorch_lightning as pl

import wandb

from src.callbacks.plotting import plot_spectrogram_reconstruction


class LogSpectrogramCallback(pl.callbacks.Callback):
    def __init__(self, num_examples=4):
        super().__init__()

        self.num_examples = num_examples
        self.outputs = []

    def on_validation_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx,
    ):
        """Called when the validation batch ends."""

        if outputs is not None:
            examples = np.min([self.num_examples, outputs["x"].shape[0]])
            self.outputs.append(outputs)

            if batch_idx == 0:
                for n in range(examples):
                    if batch_idx == 0:
                        self.log_audio(
                            outputs,
                            n,
                            trainer.logger,
                        )

    def on_validation_end(self, trainer, pl_module):
        # clear outputs
        self.outputs = []

    def log_audio(self, outputs, batch_idx, logger):
        x = outputs["x"][batch_idx, ...].float()
        x_hat = outputs["x_hat"][batch_idx, ...].float()
        curr_dafx = outputs["dafx"]

        logger.experiment.log(
            {
                f"spec/{batch_idx + 1}":
                    wandb.Image(
                        compare_spectrograms(
                            x[0:1, :],
                            x_hat[0:1, :],
                            curr_dafx,
                        )
                    )
            }
        )


def compare_spectrograms(x, x_hat, fx_name):
    image = plot_spectrogram_reconstruction(
        X=x,
        X_hat=x_hat,
        fx_name=fx_name
    )

    return image
