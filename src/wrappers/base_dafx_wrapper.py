import torch

from abc import ABC, abstractmethod


class BaseDAFXWrapper(ABC):

    @abstractmethod
    # === Public Methods ===
    def apply(self, signal: torch.Tensor, params: torch.Tensor):
        """
        Simple interface method for NN, process signal with params
        """
        pass

    @abstractmethod
    def get_num_params(self):
        """
        Returns the number of parameter values for control of DAFX
        """
        pass

    @abstractmethod
    def process_effect(self, signal: torch.Tensor):
        """
        Processes the audio given through the DAFX (with sample rate specified in constructor)
        """
        pass

    @abstractmethod
    def process_mono_as_stereo(self, signal: torch.Tensor):
        """
        Naive helper function which doubles input to stereo then converts back to mono
        """
        pass

    @abstractmethod
    def process_audio_with_random_settings(self,
                                           signal: torch.Tensor,
                                           threshold: float = 0.75,
                                           limit: int = 100,
                                           check_silence: bool = True,
                                           check_noise: bool = True
                                           ):
        pass

    @abstractmethod
    def process_audio_with_dummy_settings(self, signal: torch.Tensor):
        pass
