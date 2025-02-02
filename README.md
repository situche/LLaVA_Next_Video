# LLaVa NeXT 视频示例

此仓库展示了如何使用 **LLaVa-NeXT** 架构训练和微调一个视频-字幕生成模型。该模型基于 **ShareGPT4Video** 数据集，能够处理视频输入并生成对应的文本描述。模型的核心目标是进行视频到文本的条件生成任务，即根据视频内容生成相关的描述性文本。

该实现结合了 Hugging Face Transformers 和 PEFT（Parameter Efficient Fine-Tuning）库，同时集成了 LoRA（低秩适配）来提升训练的效率和模型的表现。通过这些技术，可以在有限的计算资源下实现大规模模型的训练。

## 依赖

- Python 3.8+
- PyTorch 1.10+
- Transformers（Hugging Face）
- Hugging Face Datasets
- Hugging Face Hub
- PEFT（参数高效微调）
- OpenCV（用于视频处理）
- NumPy（用于数据处理）
- Decord（用于视频读取）
- Matplotlib（用于结果可视化）

你可以通过以下命令安装所需的依赖：

```bash
pip install -r requirements.txt
```

## 文件概述

### 1. **模型初始化与设置**
代码从 Hugging Face Hub 加载了预训练的 `LLaVaNextVideoForConditionalGeneration` 模型，并初始化了相关的处理器和分词器。此部分的目标是设置训练和推理所需的基础组件。

```python
from transformers import LLaVaNextVideoForConditionalGeneration, LLaVaNextProcessor

# 模型和处理器初始化
model = LLaVaNextVideoForConditionalGeneration.from_pretrained('huggingface_model_checkpoint')
processor = LLaVaNextProcessor.from_pretrained('huggingface_processor_checkpoint')
```

### 2. **视频预处理**
视频数据需要被转换成适合模型输入的格式。`read_video_opencv` 函数使用 OpenCV 从视频中读取帧，并对帧进行采样。通常，视频会被切分成多个图像帧，随后将这些帧处理为张量（Tensor）以供模型训练。

```python
import cv2
import torch

def read_video_opencv(video_path, max_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return torch.tensor(frames)
```

### 3. **数据集加载与处理**
该项目使用 **ShareGPT4Video** 数据集，包含视频和相应的字幕。通过 Hugging Face 的 `datasets` 库加载数据，脚本处理这些视频并提取帧序列，格式化为模型可以接受的输入形式。

```python
from datasets import load_dataset

dataset = load_dataset('huggingface/sharegpt4video')
train_dataset = dataset['train']
```

### 4. **数据整理与填充**
由于模型的输入数据需要统一长度，`LlavaNextVideoDataCollatorWithPadding` 类用来在训练过程中对数据进行填充。这一过程确保了每个输入批次的大小一致。

```python
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

### 5. **LoRA 配置与训练**
LoRA（低秩适配）是一种用于加速大规模模型微调的技术。通过 PEFT 库，我们可以对模型进行适配和训练，优化过程中的参数变动仅限于模型的一小部分，从而减少了训练的计算负担。在训练过程中，你可以选择是否启用 LoRA 或 QLoRA（量化 LoRA）。

```python
from peft import LoraConfig, get_peft_model

# 配置 LoRA
lora_config = LoraConfig(
    r=8,   # 低秩参数
    lora_alpha=16,  # 参数放大系数
    lora_dropout=0.1,  # dropout 设置
    bias='none'
)

# 生成适配模型
model = get_peft_model(model, lora_config)
```

### 6. **训练与评估**
训练使用 Hugging Face 的 `Trainer` 类进行管理。`Trainer` 类不仅处理模型训练，还自动进行评估、保存以及检查点管理。训练过程中，可以指定评估指标、学习率调度策略等。

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.tokenizer,
    data_collator=data_collator
)

trainer.train()
```

### 7. **结果展示**
训练完成后，模型会保存并进行推理。可以使用 Matplotlib 展示生成的视频帧，帮助用户理解模型在视频-字幕生成任务上的表现。

```python
import matplotlib.pyplot as plt

# 展示生成的视频帧
def show_video_frames(video_frames):
    for frame in video_frames:
        plt.imshow(frame)
        plt.show()
```

### 8. **推送到 Hugging Face Hub**
训练完毕后，模型会被推送到 Hugging Face Hub，便于与其他人共享和部署。如果没有权限推送，可以修改 `hub_model_id` 或者使用自己的账户进行推送。

```python
model.push_to_hub("your_hub_model_name")
```

## 使用方法

### 1. **准备数据集**
下载并解压 **ShareGPT4Video** 数据集。如果你有其他格式的视频，也可以进行格式转换以支持 `.mp4`, `.avi`, `.mov` 等常见格式。

```bash
wget https://huggingface.co/datasets/sharegpt4video/raw/1.0.0/video_data.zip
unzip video_data.zip -d ./data
```

### 2. **配置训练参数**
修改代码中的训练配置，例如批量大小、学习率、最大训练步数等，确保它们适合你的硬件资源。

```python
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    logging_dir='./logs',
)
```

### 3. **启动训练**
运行训练脚本，开始模型的训练和微调。

```bash
python train.py
```

### 4. **查看训练结果**
训练完成后，可以通过生成的评估报告来查看模型的效果。你还可以查看生成的视频帧，并对模型进行调优。

```bash
python evaluate.py
```

## 注意事项

1. **视频格式**  
   目前支持的格式包括 `.mp4`, `.avi`, `.mov` 等，其他格式的视频可能需要进行格式转换。

2. **硬件要求**  
   训练过程中需要较强的计算资源，建议使用具有至少 16GB 显存的 GPU。若硬件资源不足，可考虑使用云计算平台进行训练。

3. **推送模型**  
   训练结束后，可以将模型推送至 Hugging Face Hub。如果没有权限推送，可以修改推送 ID，或使用自己的 Hugging Face 账户。

## 许可证

本项目使用 [MIT 许可证](LICENSE)，可以自由使用、修改和分发代码。
