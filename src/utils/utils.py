import numpy as np
import torch
import torchaudio

from torchaudio.transforms import MFCC


def conform_length(x: torch.Tensor, length: int):
    """Crop or pad input on last dim to match `length`."""
    if x.shape[-1] < length:
        padsize = length - x.shape[-1]
        x = torch.nn.functional.pad(x, (0, padsize))
    elif x.shape[-1] > length:
        x = x[..., :length]

    return x


def linear_fade(
        x: torch.Tensor,
        fade_ms: float = 50.0,
        sample_rate: float = 22050,
):
    """Apply fade in and fade out to last dim."""
    fade_samples = int(fade_ms * 1e-3 * sample_rate)

    fade_in = torch.linspace(0.0, 1.0, steps=fade_samples)
    fade_out = torch.linspace(1.0, 0.0, steps=fade_samples)

    # fade in
    x[..., :fade_samples] *= fade_in

    # fade out
    x[..., -fade_samples:] *= fade_out

    return x


def peak_normalise(signal: torch.Tensor):
    # peak normalize to -12 dBFS
    signal /= signal.abs().max()
    signal *= 10 ** (-12.0 / 20)  # with min 3 dBFS headroom
    return signal


def audio_to_spectrogram(signal: torch.Tensor,
                         n_fft: int = 4096,
                         hop_length: int = 2048,
                         window_size: int = 4096,
                         normalise_audio: bool = True):
    bs, _, _ = signal.size()

    if normalise_audio:
        signal = peak_normalise(signal)

    window = torch.hann_window(window_size).to(signal.device)

    X = torch.stft(
        signal.view(bs, -1),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=window_size,
        window=window,
        return_complex=True,
    )

    # Absolute value part
    X_db = torch.pow(X.abs() + 1e-8, 0.3)

    # Normalise (0,1)
    X_db_norm = (X_db - X_db.mean()) / X_db.std()

    X_db_norm = X_db_norm.unsqueeze(1).permute(0, 1, 3, 2)

    return X_db_norm


def audio_to_mel_spectrogram(signal: torch.Tensor,
                             sample_rate: int = 24_000,
                             n_mels: int = 128,
                             n_fft: int = 4096,
                             win_length: int = 1024,
                             f_max=18_000,
                             f_min=20,
                             normalise_audio: bool = True):
    bs, _, _ = signal.size()

    if normalise_audio:
        signal = peak_normalise(signal)

    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=win_length,
        f_max=f_max,
        f_min=f_min).to(signal.device)

    X = transform(signal)

    # Absolute value part
    X_db = torch.pow(X.abs() + 1e-8, 0.3)

    # # Normalise (0,1)
    X_db_norm = (X_db - X_db.mean()) / X_db.std()

    X_db_norm = X_db_norm.permute(0, 1, 3, 2)

    return X_db_norm


def is_silent(signal: torch.Tensor):
    """
    Returns True if signal is silent
    """

    signal = signal.detach().cpu().numpy()
    mean = np.square(signal).mean()

    return mean < 1e-5


def is_noise(original: torch.Tensor,
             effected: torch.Tensor,
             sample_rate: int,
             cosine_similarity_threshold: float = 0.75):
    """
    Returns True if difference between original and effected signal is too extreme.
    """
    mfcc = MFCC(sample_rate=sample_rate,
                n_mfcc=30,
                melkwargs={"n_mels": 60})

    a = mfcc(original)
    b = mfcc(effected)

    similarity = torch.nn.functional.cosine_similarity(a, b).mean()
    # print(similarity)
    return similarity < cosine_similarity_threshold


def split_dataset(file_list, subset, train_frac):
    """Given a list of files, split into train/val/test sets.

    Args:
        file_list (list): List of audio files.
        subset (str): One of "train", "val", or "test".
        train_frac (float): Fraction of the dataset to use for training.

    Returns:
        file_list (list): List of audio files corresponding to subset.
    """
    assert 0.1 < train_frac < 1.0

    total_num_examples = len(file_list)

    train_num_examples = int(total_num_examples * train_frac)
    val_num_examples = int(total_num_examples * (1 - train_frac) / 2)
    test_num_examples = total_num_examples - (train_num_examples + val_num_examples)

    if train_num_examples < 0:
        raise ValueError(
            f"No examples in training set. Try increasing train_frac: {train_frac}."
        )
    elif val_num_examples < 0:
        raise ValueError(
            f"No examples in validation set. Try decreasing train_frac: {train_frac}."
        )
    elif test_num_examples < 0:
        raise ValueError(
            f"No examples in test set. Try decreasing train_frac: {train_frac}."
        )

    if subset == "train":
        start_idx = 0
        stop_idx = train_num_examples
    elif subset == "val":
        start_idx = train_num_examples
        stop_idx = start_idx + val_num_examples
    elif subset == "test":
        start_idx = train_num_examples + val_num_examples
        stop_idx = start_idx + test_num_examples + 1
    else:
        raise ValueError("Invalid subset: {subset}.")

    return file_list[start_idx:stop_idx]


def rademacher(size):
    m = torch.distributions.binomial.Binomial(1, 0.5)
    x = m.sample(size)
    x[x == 0] = -1
    return x


def get_training_reference(x: torch.Tensor, y: torch.Tensor):
    """
    Takes an input and reference audio and splits into A/B sections.
    Randomly selects A or B for input and opposite section for reference.
    Also returns the ground truth for calculating loss
    @param x: input audio
    @param y: reference audio
    @return: x, y_ref, y (ground truth)
    """
    length = x.shape[-1]

    x_A = x[..., : length // 2]
    x_B = x[..., length // 2:]

    y_A = y[..., : length // 2]
    y_B = y[..., length // 2:]

    if torch.rand(1).sum() > 0.5:
        y_ref = y_B
        y = y_A
        x = x_A
    else:
        y_ref = y_A
        y = y_B
        x = x_B

    return x, y_ref, y


def effect_to_end_to_end_checkpoint_id(effect_name):
    chkpt_map = {
        "overdrive": "fboro0y2",
        "ringmod": "c5rp55l2",
        "delay": "gg4q2yj9",
        "combo": "8283y9mm",
        "multiband": "th24l5fs",
        "ambience": "6d7hvfwc",
        "dynamics": "ql2ll6wp",
        "flanger": "d2yk0n9l",
        "leslie": "znm7qcsc",
    }

    if len(effect_name.split()) >= 2:
        effect_name = effect_name.split()[-1]

    effect_name = effect_name.lower()

    return chkpt_map.get(effect_name)

