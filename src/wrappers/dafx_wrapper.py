import librosa
import numpy as np
import torch

from src.wrappers.base_dafx_wrapper import BaseDAFXWrapper

import src.utils.utils as utils


class DAFXWrapper(BaseDAFXWrapper):
    def __init__(self, dafx, sample_rate, param_names=None):
        """
        Wrapper for Pedalboard VST3 to add useful helper methods
        @param dafx: A Pedalboard VST3 plugin
        @param sample_rate: Sample rate for audio to be processed
        @param param_names: An optional list of specified parameters, if None will use all parameters
        """
        self.dafx = dafx
        self.sample_rate = sample_rate
        self.param_names = param_names

        self.param_min_max_vals = None
        self.idx_to_param_map = None
        self.param_to_idx_map = None

        self._build_attributes()

    # === Private Methods ===
    def _assert_valid_vector_length(self, vector):
        """
        Assert that length of given vector matches number of DAFX params
        """
        assert (len(vector) == self.get_num_params())

    def _build_attributes(self):
        self.param_names, self.param_min_max_vals = self._initialise_parameter_attributes()
        self.idx_to_param_map = {i: self.param_names[i] for i in range(self.get_num_params())}
        self.param_to_idx_map = {v: k for k, v in self.idx_to_param_map.items()}

    def _denormalise(self, param_name, param_value):
        """
        Transform normalised parameter value to raw parameter value
        """
        param_min = self.param_min_max_vals[param_name]['min']
        param_max = self.param_min_max_vals[param_name]['max']
        return param_value * (param_max - param_min) + param_min

    def _filter_param_names(self, all_parameters):
        """
        Check list of parameter names given are valid and filter those not required
        """
        if self.param_names is None:
            return all_parameters
        # use specified parameter names
        out_params = [p for p in all_parameters if p in self.param_names]
        if len(out_params) == len(self.param_names):
            return out_params

        bad_vals = [p for p in self.param_names if p not in all_parameters]
        raise ValueError(f"DAFX does not have parameters: {bad_vals}.")

    def _get_name_from_index(self, idx):
        return self.idx_to_param_map[idx]

    def _initialise_parameter_attributes(self):
        """
        Returns adjustable parameter names and their min/max values
        """
        all_params = list(self.dafx.parameters.keys())
        filtered_params = self._filter_param_names(all_params)

        min_max_map = {}
        for p in filtered_params:
            param = self.dafx.parameters[p]
            if self._parameter_is_adjustable(param):
                min_max_map[p] = {'min': param.min_value, 'max': param.max_value}

        return list(min_max_map.keys()), min_max_map

    def _normalise(self, param_name, param_value):
        """
        Take a parameter value and normalise it
        """
        param_min = self.param_min_max_vals[param_name]['min']
        param_max = self.param_min_max_vals[param_name]['max']
        return (param_value - param_min) / (param_max - param_min)

    def _parameter_is_adjustable(self, parameter):
        return not self._parameter_is_boolean_or_none(parameter)

    @staticmethod
    def _safe_convert_tensor_to_numpy_cpu(x):
        if torch.is_tensor(x):
            return x.cpu().numpy()
        return x

    @staticmethod
    def _parameter_is_boolean_or_none(parameter):
        return isinstance(parameter.range[0], bool) or isinstance(parameter.range[1], bool) \
               or parameter.range[0] is None or parameter.range[1] is None

    def _set_attribute(self, name, value):
        setattr(self.dafx, name, value)

    # === Public Methods ===
    def apply(self, signal, params):
        """
        Simple interface method for NN
        """
        signal = self._safe_convert_tensor_to_numpy_cpu(signal)
        params = self._safe_convert_tensor_to_numpy_cpu(params)
        params = self._clip_params(params)

        # Set parameters
        self.set_normalised_parameter_vector(params)

        # Process
        if len(signal.shape) == 1 or signal.shape[0] == 1:
            signal = signal.squeeze()
            effected = self.process_mono_as_stereo(signal)
        else:
            effected = self.process_effect(signal)

        return torch.Tensor(effected)

    def get_current_param_settings(self):
        """
        Returns parameter names and their denormalised values
        """
        param_dict = {}
        for param in self.param_names:
            param_dict[param] = getattr(self.dafx, param)
        return param_dict

    def get_current_normalised_param_settings(self):
        """
        Returns parameter names and their denormalised values
        """
        param_dict = {}
        for param in self.param_names:
            raw = getattr(self.dafx, param)
            param_dict[param] = self._normalise(param, raw)
        return param_dict

    def get_dummy_parameter_settings(self):
        params_vec_size = self.get_num_params()
        return torch.ones(params_vec_size) / 2

    def get_num_params(self):
        """
        Returns the number of parameter values for control of DAFX
        """
        return len(self.param_names)

    def get_random_parameter_settings(self):
        param_vec_size = self.get_num_params()
        return torch.rand(param_vec_size)

    def process_effect(self, signal):
        """
        Processes the audio given through the DAFX (with sample rate specified in constructor)
        """
        x = self._safe_convert_tensor_to_numpy_cpu(signal)
        return self.dafx(x, sample_rate=self.sample_rate)

    def process_mono_as_stereo(self, signal):
        """
        Naive helper function which doubles input to stereo then converts back to mono
        """
        y = self._safe_convert_tensor_to_numpy_cpu(signal)
        y = np.array([y, y])
        effected = self.process_effect(y)
        return librosa.to_mono(effected)

    def process_audio_with_random_settings(self,
                                           signal: torch.Tensor,
                                           threshold: float = 0.75,
                                           limit: int = 100,
                                           check_silence: bool = True,
                                           check_noise: bool = True
                                           ):
        effect_count = 0
        while True:
            random_setting = self.get_random_parameter_settings()
            effected = self.apply(signal, random_setting)

            silent = utils.is_silent(effected) if check_silence else False
            noise = utils.is_noise(signal, effected, self.sample_rate) if check_noise else False

            if not silent and not noise:
                return effected

            effect_count += 1
            if effect_count > limit:
                print("[process_with_random_settings]: Effect limit exceeded, returning original signal")
                return signal

    def process_audio_with_dummy_settings(self, signal: torch.Tensor):
        dummy_setting = self.get_dummy_parameter_settings()
        return self.apply(signal, dummy_setting)

    def set_normalised_index_parameter(self, idx, value):
        """
        Apply a normalised value to a parameter by index
        """
        param_name = self._get_name_from_index(idx)
        self.set_normalised_named_parameter(param_name, value)

    def set_normalised_named_parameter(self, name, value):
        """
        Apply a normalised value to a parameter by name
        """
        raw_val = self._denormalise(name, value)
        self.set_denormalised_named_parameter(name, raw_val)

    def set_normalised_parameter_vector(self, vector):
        """
        Apply a vector of normalised values to all parameters
        """
        self._assert_valid_vector_length(vector)
        for idx in range(len(vector)):
            self.set_normalised_index_parameter(idx, vector[idx])

    def set_denormalised_index_parameter(self, idx, value):
        """
        Apply a denormalised value to a parameter by index
        """
        param_name = self._get_name_from_index(idx)
        self.set_denormalised_named_parameter(param_name, value)

    def set_denormalised_named_parameter(self, name, value):
        """
        Apply a denormalised value to a parameter by name
        """
        self._set_attribute(name, value)

    def set_denormalised_parameter_vector(self, vector):
        """
        Apply a vector of denormalised values to all parameters
        """
        self._assert_valid_vector_length(vector)

        for idx in range(len(vector)):
            self.set_denormalised_index_parameter(idx, vector[idx])

    @staticmethod
    def _clip_params(params):
        return np.clip(params, a_min=0.01, a_max=0.99)
