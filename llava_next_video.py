import os
import av
import fsspec
import shutil
import numpy as np

from transformers import Trainer, TrainingArguments, Seq2SeqTrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

import torch
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import snapshot_download, hf_hub_download, HfFileSystem
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict

import cv2
from numba import jit, cuda
from decord import VideoReader, gpu, cpu

from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML

MAX_LENGTH = 256
BATCH_SIZE = 4
NUM_FRAMES = 8
DATASET_PATH = 'path'
OUTPUT_DIR = 'path'
MODEL_ID = 'llava-hf/LLaVa-NeXT-Video-7b-hf'
REPO_ID = 'RaushanTurganbay/LLaVa-NeXT-Video-demo'

USE_LORA = False
USE_QLORA = True

dataset = load_dataset("ShareGPT4Video/ShareGPT4Video")

def read_video_opencv(video_path, num_frames=NUM_FRAMES):
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.arange(0, total_num_frames, total_num_frames / num_frames).astype(int)
    frames = process_video_cv2(video, indices, total_num_frames)
    return np.stack(frames)

def process_video_cv2(video: cv2.VideoCapture, indices: np.array, length: int):
    index = 0 
    frames = [] 
    while video.isOpened():  
        success, frame = video.read()
        if index in indices:
            height, width, channel = frame.shape
            frames.append(frame[0:height, 0:width, 0:channel])
        if success:
            index += 1
        if index >= length:
            break
    
    video.release() 
    return frames

processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
processor.tokenizer.padding_side = 'right'

def collate_fn(example, path): 
    video_file = example['video_path']。split('/')[-1]
    video_clip = read_video_decord(f'{path}/{video_file}')

    captions_all = [caption for caption in example['captions'] if caption['idx'] == '-1']
    caption = captions_all[0]['content'] 

    conversation = [ 
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'Provide a detailes caption for this video.'},
                {'type': 'video'}
            ]
        },
        {
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': caption}
            ]
        }
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_ptompt=False) 

    batch = processor(
        text=prompt,
        videos=video_clip,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    return batch

def process_dataset(zip_file_path):
    zip_folder = os.path.basename(zip_file_path).replace('.zip', '')
    data_path = f"{DATASET_PATH}/videos_ShareGPT/{zip_folder}"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Directory {data_path} does not exist. Did you unzip the files correctly?")

    video_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')): 
                video_files.append({"video_path": os.path.join(root, file)})

    if not video_files:
        raise ValueError(f"No video files found in {data_path}. Check the zip folder content.")

    dataset = Dataset.from_list(video_files)

    dataset = dataset.map(
        lambda example: {"processed_video_path": example["video_path"]},
        num_proc=8  # Parallel processing
    )

    return DatasetDict({"train": dataset})

datasets_combined = []
fs = HfFileSystem()
directory = f'{DATASET_PATH}/temp_dir' 

zip_folders = {'mixit', 'bdd100k', 'ego4d', 'pexels', 'pixabay'}
for zip_folder in zip_folders:
    print(f'Processing folders: {zip_folder}')
    zip_files = fs.ls(f'datasets/ShareGPT4Video/ShareGPT4Video/zip_folder/{zip_folder}', detail=False) 
    for zip_file in zip_files: 
        zip_file.split('/')[-1]    
        path = hf_hub_download( 
            repo_id='ShareGPT4Video/ShareGPT4Video',
            repo_type='dataset',
            filename=f'zip_folder/{zip_folder}/{zip_file}',
            local_dir=f'{DATASET_PATH}/{zip_folder}',
            cache_dir=DATASET_PATH
    )
    subdataset_name = zip_file.split('_')[0]

    if os.path.exists(directory): 
        shutil.rmtree(directory)
    os.makedirs(directory)

    if path.endswith('.zip'):
        shutil.unpack_archive(path, directory)

        curr_video_files = os.listdir(directory) 
        small_dataset = dataset.filter(lambda example: example['video_path'].split('/')[-1] in curr_video_files) 

        small_dataset = small_dataset.map( 
            collate_fn,
            batched=False,
            fn_kwargs={'path': directory},
            num_proc=2,
            remove_columns=['captions', 'keyframe', 'timestamp', 'video_id', 'video_path'],
            writer_batch_size=400
        )
        datasets_combined.append(small_dataset['train'])
    os.remove(path) 

video_path = snapshot_download(repo_id='ShareGPT4Video/ShareGPT4Video', repo_type="dataset", allow_patterns="*videos.zip")

datasets_combined = []
directory = f'{DATASET_PATH}/videos_ShareGPT/'
zip_folders = {'ego4d', 'mixit', 'pixabay', 'bdd100k'}

for zip_folder in zip_folders:
    for zip_file in os.listdir(f'{video_path}/{zip_folder}'):
        zip_file_path = f'{video_path}/{zip_folder}/{zip_file}'
        shutil.unpack_archive(path, f'{directory}/{zip_folder}')
    
    small_dataset = dataset.filter(lambda example: example['video_path'].startswith(zip_folder))
    small_dataset = small_dataset.map(collate_fn, batched=False, fn_kwargs={'path': f'{directory}/{zip_folder}'}, num_proc=8)
    temp_dataset = process_dataset(zip_file) 
    datasets_combined.append(temp_dataset['train'])

dataset_processed = concatenate_datasets(datasets_combined)
dataset_processed = dataset_processed.shuffle(seed=42)
dataset = dataset_processed.train_test_split(test_size=0.2) 

train_dataset, test_dataset = dataset['train'].with_format('torch'), dataset['test'].with_format('torch')
print(train_dataset, test_dataset)

class LlavaNextVideoDataCollatorWithPadding: 
    def __init__(self, processor):
        self.processor = processor 

    def __call__(self, features):
        padded_inputs = self.processor.tokenizer.pad( 
            {
                'input_ids': [feat['input_ids'][0] for feat in features],
                'attention_mask': [feat['attention_mask'][0] for feat in features]
            },
            padding=True,
            return_tensors='pt'
        )

        labels = padded_inputs['input_ids'].clone()  
        labels[labels == self.processor.tokenizer.pad_token_id] = -100  
        padded_inputs['labels'] = labels 
        padded_inputs['pixel_values_videos'] = torch.cat([feat['pixel_values_videos'] for feat in features], dim=0)

        return padded_inputs 

example = train_dataset[0] 

clip = example['pixel_values_videos'][0] * 255
clip = clip.permute(0, 2, 3, 1).clamp(0, 255)

video = np.array(clip).astype(np.uint8) 

fig = plt.figure() 
im = plt.imshow(video[0, :, :, :]) 
plt.close() 

def init():
    im.set_data(video[0,:,:,:])

def animate(i):
    im.set_data(video[i,:,:,:])
    return im

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=video.shape[0], interval=100)
print(HTML(anim.to_html5_video()))

if USE_LORA or USE_QLORA:
    if USE_QLORA:  
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype='nf4',
            bnb_4bit_quant_dtype=torch.float16
        )
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map='auto'
    )
else: 
    model=LlavaNextVideoForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        _attn_implementation='flash_attention_2',
        device_map='auto'
    )

def find_all_linear_names(model): 
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodel_keywords = ['multi_model_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodel_keywords):
            continue
        if isinstance(module, cls): 
            names = name.split('.')  
            lora_module_names.add(names[0] if len(names) == 1 else names[-1]) 
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

lora_config = LoraConfig( 
    r=8, 
    lora_alpha=8,  
    lora_dropout=0.1,  
    target_modules=find_all_linear_names(model), 
    init_lora_weights='gaussian'
)

model = prepare_model_for_kbit_training(model) 
model = get_peft_model(model, lora_config)
print(model)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy='steps', 
    eval_steps=20, 
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=8, 
    learning_rate=2e-05,
    max_steps=100, 
    lr_scheduler_type='cosine', 
    warmup_ratio=0.1,
    logging_steps=20,
    save_strategy='steps',
    save_steps=20,
    save_total_limit=1, 
    fp16=True,
    fp16_full_eval=True, 
    optim='adamw_bnb_8bit', 
    hub_model_id='wandb',  
    push_tO_hub=True, 
    label_names=['labels'],  
    dataloader_num_workers=4 
)

trainer = Trainer(
    model=model, 
    args=args,
    train_dataset=train_dataset, 
    eval_dataset=test_dataset, 
    data_collator=LlavaNextVideoDataCollatorWithPadding(processor=processor) 
)

print(trainer.train())
print(trainer.model.push_to_hub(REPO_ID))

model = LlavaNextVideoForConditionalGeneration.from_pretrained( 
    REPO_ID,
    torch_dtype = torch.float16,
    device_map='auto'
)

example = test_dataset[0]
clip = example['pixel_values_videos'][0] * 225  
clip = clip.permute(0, 2, 3, 1).clamp(0, 255)

video = np.array(clip).astype(np.uint8) 

fig = plt.figure()
im = plt.imshow(video[0,:,:,:])
plt.close()

def init():
    im.set_data(video[0,:,:,:])

def animate(i):
    im.set_data(video[i,:,:,:])
    return im

anim = animate.FuncAnimation(fig, animate, init_func=init, frames=video.shape[0], interval=100)
print(HTML(anim.to_html5_video())) 
