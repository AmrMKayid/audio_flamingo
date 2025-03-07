import os
import yaml
import json
import torch
import spaces
import librosa
import argparse
import numpy as np
import gradio as gr
from tqdm import tqdm
import soundfile as sf
from pydub import AudioSegment
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

from data.data import get_audiotext_dataloader
from src.factory import create_model_and_transforms
from train.train_utils import Dict2Class, get_autocast, get_cast_dtype

HEADER = ("""
<div style="display: flex; justify-content: center; align-items: center; text-align: center;">
  <a href="https://github.com/NVIDIA/audio-flamingo" style="margin-right: 20px; text-decoration: none; display: flex; align-items: center;">
    <img src="https://github.com/NVIDIA/audio-flamingo/blob/main/assets/af_logo.png?raw=true" alt="Audio Flamingo 2 🔥🚀🔥" style="max-width: 120px; height: auto;">
  </a>
  <div>
    <h1>Audio Flamingo 2: An Audio-Language Model with Long-Audio Understanding and Expert Reasoning Abilities</h1>
    <h5 style="margin: 0;">If this demo please you, please give us a star ⭐ on Github or 💖 on this space.</h5>
  </div>
</div>

<div style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://github.com/NVIDIA/audio-flamingo"><img src='https://img.shields.io/badge/Github-AudioFlamingo2-9C276A' style="margin-right: 5px;"></a>
  <a href="https://arxiv.org/abs/2503.03983"><img src="https://img.shields.io/badge/Arxiv-2503.03983-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://huggingface.co/nvidia/audio-flamingo-2"><img src="https://img.shields.io/badge/🤗-Checkpoints-ED5A22.svg" style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/audio-flamingo/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/audio-flamingo.svg?style=social"></a>
</div>
""")

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

api_key = os.getenv("my_secret")

snapshot_download(repo_id="nvidia/audio-flamingo-2", local_dir="./", token=api_key)

config = yaml.load(open("configs/inference.yaml"), Loader=yaml.FullLoader)

data_config = config['data_config']
model_config = config['model_config']
clap_config = config['clap_config']
args = Dict2Class(config['train_config'])

model, tokenizer = create_model_and_transforms(
    **model_config,
    clap_config=clap_config, 
    use_local_files=args.offline,
    gradient_checkpointing=args.gradient_checkpointing,
    freeze_lm_embeddings=args.freeze_lm_embeddings,
)

device_id = 0
model = model.to(device_id)
model.eval()

# Load metadata
with open("safe_ckpt/metadata.json", "r") as f:
    metadata = json.load(f)

# Reconstruct the full state_dict
state_dict = {}

# Load each SafeTensors chunk
for chunk_name in metadata:
    chunk_path = f"safe_ckpt/{chunk_name}.safetensors"
    chunk_tensors = load_file(chunk_path)

    # Merge tensors into state_dict
    state_dict.update(chunk_tensors)

x,y = model.load_state_dict(state_dict, False)

autocast = get_autocast(
    args.precision, cache_enabled=(not args.fsdp)
)

cast_dtype = get_cast_dtype(args.precision)

def get_num_windows(T, sr):

    window_length  = int(float(clap_config["window_length"]) * sr)
    window_overlap = int(float(clap_config["window_overlap"]) * sr)
    max_num_window = int(clap_config["max_num_window"])

    num_windows = 1
    if T <= window_length:
        num_windows = 1
        full_length = window_length
    elif T >= (max_num_window * window_length - (max_num_window - 1) * window_overlap):
        num_windows = max_num_window
        full_length = (max_num_window * window_length - (max_num_window - 1) * window_overlap)
    else:
        num_windows = 1 + int(np.ceil((T - window_length) / float(window_length - window_overlap)))
        full_length = num_windows * window_length - (num_windows - 1) * window_overlap
    
    return num_windows, full_length


def read_audio(file_path, target_sr=16000, duration=30.0, start=0.0):

    if file_path.endswith('.mp3'):
        audio = AudioSegment.from_file(file_path)
        if len(audio) > (start + duration) * 1000:
            audio = audio[start * 1000:(start + duration) * 1000]

        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)

        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        data = np.array(audio.get_array_of_samples())
        if audio.sample_width == 2:
            data = data.astype(np.float32) / np.iinfo(np.int16).max
        elif audio.sample_width == 4:
            data = data.astype(np.float32) / np.iinfo(np.int32).max
        else:
            raise ValueError("Unsupported bit depth: {}".format(audio.sample_width))

    else:
        with sf.SoundFile(file_path) as audio:
            original_sr = audio.samplerate
            channels = audio.channels

            max_frames = int((start + duration) * original_sr)

            audio.seek(int(start * original_sr))
            frames_to_read = min(max_frames, len(audio))
            data = audio.read(frames_to_read)

            if data.max() > 1 or data.min() < -1:
                data = data / max(abs(data.max()), abs(data.min()))
        
        if original_sr != target_sr:
            if channels == 1:
                data = librosa.resample(data.flatten(), orig_sr=original_sr, target_sr=target_sr)
            else:
                data = librosa.resample(data.T, orig_sr=original_sr, target_sr=target_sr)[0]
        else:
            if channels != 1:
                data = data.T[0]
    
    if data.min() >= 0:
        data = 2 * data / abs(data.max()) - 1.0
    else:
        data = data / max(abs(data.max()), abs(data.min()))
    
    assert len(data.shape) == 1, data.shape
    return data

def load_audio(audio_path):

    sr = 16000
    window_length  = int(float(clap_config["window_length"]) * sr)
    window_overlap = int(float(clap_config["window_overlap"]) * sr)
    max_num_window = int(clap_config["max_num_window"])
    duration = max_num_window * (clap_config["window_length"] - clap_config["window_overlap"]) + clap_config["window_overlap"]

    audio_data = read_audio(audio_path, sr, duration, 0.0) # hard code audio start to 0.0
    T = len(audio_data)
    num_windows, full_length = get_num_windows(T, sr)

    # pads to the nearest multiple of window_length
    if full_length > T:
        audio_data = np.append(audio_data, np.zeros(full_length - T))

    audio_data = audio_data.reshape(1, -1)
    audio_data_tensor = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()

    audio_clips = []
    audio_embed_mask = torch.ones(num_windows)
    for i in range(num_windows):
        start = i * (window_length - window_overlap)
        audio_data_tensor_this = audio_data_tensor[:, start:start+window_length]
        audio_clips.append(audio_data_tensor_this)

    if len(audio_clips) < max_num_window:
        audio_clips = audio_clips[:max_num_window]
        audio_embed_mask = audio_embed_mask[:max_num_window]

    audio_clips = torch.cat(audio_clips)
    
    return audio_clips, audio_embed_mask

@spaces.GPU
def predict(filepath, question):

    audio_clips, audio_embed_mask = load_audio(filepath)
    audio_clips = audio_clips.to(device_id, dtype=cast_dtype, non_blocking=True)
    audio_embed_mask = audio_embed_mask.to(device_id, dtype=cast_dtype, non_blocking=True)

    text_prompt = str(question).lower()
    text_output = str(question).lower()

    sample = f"<audio>{text_prompt.strip()}{tokenizer.sep_token}"
    # None<|endofchunk|>{tokenizer.eos_token}"

    text = tokenizer(
        sample,
        max_length=512,
        padding="longest",
        truncation="only_first",
        return_tensors="pt"
    )

    input_ids = text["input_ids"].to(device_id, non_blocking=True)

    media_token_id = tokenizer.encode("<audio>")[-1]
    sep_token_id = tokenizer.sep_token_id

    prompt = input_ids

    with torch.no_grad():
        output = model.generate(
            audio_x=audio_clips.unsqueeze(0),
            audio_x_mask=audio_embed_mask.unsqueeze(0),
            lang_x=prompt,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=256,
            temperature=0.0)[0]
    
    output_decoded = tokenizer.decode(output).split(tokenizer.sep_token)[-1].replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').replace('<|endofchunk|>', '')

    return output_decoded

audio_examples = [
    ["./examples/emergent1.wav", "What is unusual about the audio?"],
    ["./examples/soundcap1.wav", "What is the soundscape in this audio?"],
    ["./examples/muscicap1.wav", "Summarize the music content in a sentence."],
    ["./examples/mmau1.wav", "What specific sounds can be distinguished from the audio clip? (A) Helicopter and impact sounds  (B) Whistling and chatter (C) Car honking and raindrops (D) Birds chirping and water flowing"],
]


demo = gr.Blocks()
with demo:
    gr.HTML(HEADER)
    gr.Interface(fn=predict,
                    inputs=[gr.Audio(type="filepath"), gr.Textbox(value='Describe the audio.', label='Question')],
                    outputs=[gr.Textbox(label="Audio Flamingo 2 Output")],
                    cache_examples=True,
                    examples=audio_examples,
                    title="Audio Flamingo 2 Demo",
                    description="Audio Flamingo 2 is NVIDIA's latest Large Audio-Language Model that is capable of understanding audio inputs and answer any open-ended question about it. <br>" +
                    "**Audio Flamingo 2 is not an ASR model and has limited ability to recognize the speech content. It primarily focuses on perception and understanding of non-speech sounds and music.**<br>" +
                    "The demo is hosted on the Stage 2 checkpoints and supports upto 90 seconds of audios. Stage 3 checkpoints that support upto 5 minutes will be released at a later point.")
demo.launch(share=True)