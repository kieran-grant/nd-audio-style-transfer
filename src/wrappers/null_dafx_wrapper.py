import torch

from src.wrappers.base_dafx_wrapper import BaseDAFXWrapper


class NullDAFXWrapper(BaseDAFXWrapper):

    def apply(self, signal: torch.Tensor, params: torch.Tensor):
        return signal

    def get_num_params(self):
        return 0

    def process_effect(self, signal: torch.Tensor):
        return signal

    def process_mono_as_stereo(self, signal: torch.Tensor):
        return signal

    def process_audio_with_random_settings(self, signal: torch.Tensor, threshold: float = 0.75, limit: int = 100,
                                           check_silence: bool = True, check_noise: bool = True):
        return signal

    def process_audio_with_dummy_settings(self, signal: torch.Tensor):
        return signal
