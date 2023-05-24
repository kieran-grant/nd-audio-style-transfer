# Style Transfer for Non-differentiable Audio Effects

## Abstract

*Digital audio effects are widely used by audio engineers to alter the acoustic and temporal qualities of audio data. However, these effects can have a large number of parameters which can make them difficult to learn for beginners and hamper creativity for professionals. Recently, there have been a number of efforts to employ progress in deep learning to acquire the low-level parameter configurations of audio effects by minimising an objective function between an input and reference track, commonly referred to as style transfer. However, current approaches use inflexible black-box techniques or require that the effects under consideration are implemented in an auto-differentiation framework. In this work, we propose a deep learning approach to audio production style matching which can be used with effects implemented in some of the most widely used frameworks, requiring only that the parameters under consideration have a continuous domain. Further, our method includes style matching for various classes of effects, many of which are difficult or impossible to be approximated closely using differentiable functions. We show that our audio embedding approach creates logical encodings of timbral information, which can be used for a number of downstream tasks. Further, we perform a listening test which demonstrates that our approach is able to convincingly style match a compression/equalisation effect.*

## Style Matching Examples

You can find examples of style matching performance here: https://nondiff-style-transfer.streamlit.app/

## How to run this project

### (Optional) Download pretrained weights

> Available here: https://drive.google.com/drive/folders/1shTUBHd3CHCOEzPT_NZo3T2LOsaqXZW7?usp=sharing

### 1. Download and build mda-vst plugins

> Available here: https://github.com/elk-audio/mda-vst3

### 2. Download audio datasets

> `python src/utils/download.py --datasets [daps, vctk, musdb] --download --process`

### 3. Train Spectrogram beta-VAE

> `python src/train_scripts/train_spectrogram_vae.py --audio_dir <path/to/audio_dir> --dafx_file <path/to/mda.vst3>`

### 4. Train End-to-end Network

> `python src/train_scripts/train_end_to_end.py --audio_dir <path/to/audio_dir> --dafx_file <path/to/mda.vst3> --audio_encoder_ckpt <path/to/spectrogram.ckpt> `
