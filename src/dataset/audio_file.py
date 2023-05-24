# Adapted from:
# https://github.com/adobe-research/DeepAFx-ST

import warnings

import pyloudnorm as pyln
import torchaudio


class AudioFile(object):
    def __init__(self, filepath, preload=False, half=False, target_loudness=None):
        """Base class for audio files to handle metadata and loading.

        Args:
            filepath (str): Path to audio file to load from disk.
            preload (bool, optional): If set, load audio data into RAM. Default: False
            half (bool, optional): If set, store audio data as float16 to save space. Default: False
            target_loudness (float, optional): Loudness normalize to dB LUFS value. Default:
        """
        super().__init__()

        self.filepath = filepath
        self.half = half
        self.target_loudness = target_loudness
        self.loaded = False

        if preload:
            self.load()
            num_frames = self.audio.shape[-1]
            num_channels = self.audio.shape[0]
        else:
            metadata = torchaudio.info(filepath)
            audio = None
            self.sample_rate = metadata.sample_rate
            num_frames = metadata.num_frames
            num_channels = metadata.num_channels

        self.num_frames = num_frames
        self.num_channels = num_channels

    def load(self):
        audio, sr = torchaudio.load(self.filepath, normalize=True)
        self.audio = audio
        self.sample_rate = sr

        if self.target_loudness is not None:
            self.loudness_normalize()

        if self.half:
            self.audio = audio.half()

        self.loaded = True

    def loudness_normalize(self):
        meter = pyln.Meter(self.sample_rate)

        # convert mono to stereo
        if self.audio.shape[0] == 1:
            tmp_audio = self.audio.repeat(2, 1)
        else:
            tmp_audio = self.audio

        # measure integrated loudness
        input_loudness = meter.integrated_loudness(tmp_audio.numpy().T)

        # compute and apply gain
        gain_dB = self.target_loudness - input_loudness
        gain_ln = 10 ** (gain_dB / 20.0)
        self.audio *= gain_ln

        # check for potentially clipped samples
        if self.audio.abs().max() >= 1.0:
            warnings.warn("Possible clipped samples in output.")
