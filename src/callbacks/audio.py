# Adapted from:
# https://github.com/adobe-research/DeepAFx-ST

import auraloss
import numpy as np
import pytorch_lightning as pl

import wandb

from src.callbacks.plotting import plot_multi_spectrum
from src.callbacks.metrics import (
    LoudnessError,
    SpectralCentroidError,
    CrestFactorError,
    PESQ,
    MelSpectralDistance,
)


class LogAudioCallback(pl.callbacks.Callback):
    def __init__(self, num_examples=4, peak_normalize=True, sample_rate=24_000):
        super().__init__()
        self.num_examples = 4
        self.peak_normalize = peak_normalize

        self.metrics = {
            "PESQ": PESQ(sample_rate),
            "MRSTFT": auraloss.freq.MultiResolutionSTFTLoss(
                fft_sizes=[32, 128, 512, 2048, 8192, 32768],
                hop_sizes=[16, 64, 256, 1024, 4096, 16384],
                win_lengths=[32, 128, 512, 2048, 8192, 32768],
                w_sc=0.0,
                w_phs=0.0,
                w_lin_mag=1.0,
                w_log_mag=1.0,
            ),
            "MSD": MelSpectralDistance(sample_rate),
            "SCE": SpectralCentroidError(sample_rate),
            "CFE": CrestFactorError(),
            "LUFS": LoudnessError(sample_rate),
        }

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
                            pl_module.hparams.sample_rate,
                            pl_module.hparams.val_length,
                            trainer.global_step,
                            trainer.logger,
                        )

    def on_validation_end(self, trainer, pl_module):
        metrics = {
            "PESQ": [],
            "MRSTFT": [],
            "MSD": [],
            "SCE": [],
            "CFE": [],
            "LUFS": [],
        }
        for output in self.outputs:
            for metric_name, metric in self.metrics.items():
                try:
                    val = metric(output["y_hat"], output["y"])
                    metrics[metric_name].append(val)
                except:
                    pass

        # log final mean metrics
        for metric_name, metric in metrics.items():
            if len(metric) == 0:
                continue

            val = np.mean(metric)
            trainer.logger.experiment.log(
                {
                    f"metrics/{metric_name}": val
                }
            )

        # clear outputs
        self.outputs = []

    def compute_metrics(self, metrics_dict, outputs, batch_idx, global_step):
        # extract audio
        y = outputs["y"][batch_idx, ...].float()
        y_hat = outputs["y_hat"][batch_idx, ...].float()

        # compute all metrics
        for metric_name, metric in self.metrics.items():
            try:
                val = metric(y_hat.view(1, 1, -1), y.view(1, 1, -1))
                metrics_dict[metric_name].append(val)
            except:
                pass

    def log_audio(self, outputs, batch_idx, sample_rate, n_fft, global_step, logger):
        p = outputs["p"][batch_idx, ...].float()
        x = outputs["x"][batch_idx, ...].float()
        y = outputs["y"][batch_idx, ...].float()
        y_hat = outputs["y_hat"][batch_idx, ...].float()
        y_ref = outputs["y_ref"][batch_idx, ...].float()
        param_lookup = outputs["param_names"]

        for idx in range(len(p)):
            logger.experiment.log(
                {
                    f"dafx_parameters/{param_lookup[idx]}":
                        p[idx].squeeze()
                }
            )

        if self.peak_normalize:
            x /= x.abs().max()
            y /= y.abs().max()
            y_hat /= y_hat.abs().max()
            y_ref /= y_ref.abs().max()

        logger.experiment.log(
            {
                f"x/{batch_idx + 1}":
                    wandb.Audio(x[0:1, :].squeeze(),
                                sample_rate=sample_rate)
            }
        )

        logger.experiment.log(
            {
                f"y/{batch_idx + 1}":
                    wandb.Audio(y[0:1, :].squeeze(),
                                sample_rate=sample_rate)
            }
        )

        logger.experiment.log(
            {
                f"y_hat/{batch_idx + 1}":
                    wandb.Audio(y_hat[0:1, :].squeeze(),
                                sample_rate=sample_rate)
            }
        )

        logger.experiment.log(
            {
                f"y_ref/{batch_idx + 1}":
                    wandb.Audio(y_ref[0:1, :].squeeze(),
                                sample_rate=sample_rate)
            },
        )

        logger.experiment.log(
            {
                f"spec/{batch_idx + 1}":
                    wandb.Image(
                        compare_spectra(
                            y_hat[0:1, :],
                            y[0:1, :],
                            x[0:1, :],
                            baseline_y_hat=y_ref[0:1, :],
                            sample_rate=sample_rate,
                            n_fft=n_fft
                        )
                    )
            }
        )


def compare_spectra(
        y_hat, y, x, baseline_y_hat=None, sample_rate=44100, n_fft=16384
):
    legend = ["Input"]
    signals = [x]

    if baseline_y_hat is not None:
        legend.append("Reference")
        signals.append(baseline_y_hat)

    legend.append("Model Prediction")
    signals.append(y_hat)
    legend.append("True")
    signals.append(y)

    image = plot_multi_spectrum(
        ys=signals,
        legend=legend,
        sample_rate=sample_rate,
        n_fft=n_fft,
    )

    return image
