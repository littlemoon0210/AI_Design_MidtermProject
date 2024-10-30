# AI_Design_Project

## Project Overview
This project involves training a language model for AI-Design Course(SME637003.01) in Fudan University.
NanoGPT is a minimalistic implementation of the GPT (Generative Pre-trained Transformer) model, designed for educational purposes and small-scale experiments. This project aims to provide a clear and concise codebase for understanding the inner workings of GPT models.

Group Members
| Name   | Student Number |
| ------ | -------------- |
| 柯小月 | 24212020097   |
| 任钰浩 | 24112020153    |
## Features

- Simple and easy-to-understand code
- Minimal dependencies
- Suitable for small-scale experiments and educational purposes

## Installation

To install the necessary dependencies, run:

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```
Denpendencies:
- [pytorch](https://pytorch.org) < 3
- [numpy](https://numpy.org/install/) < 3
- transformers for huggingface transformers <  3(to load GPT-2 checkpoints)
- datasets for huggingface datasets < 3 (if you want to download + preprocess OpenWebText)
- tiktoken for OpenAI's fast BPE code < 3
- wandb for optional logging < 3
- tqdm for progress bars < 3

## Quick Start
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

First, navigate to the folder where you keep your projects and clone this repository to this folder:

```bash
git clone git@github.com:littlemoon0210/AI_Design_MidtermProject.git
```

Then, open the repository folder:

```bash
cd AI_Design_MidtermProject
```
Now, let's start training nanoGPT step by step. Run `downloadDataset.py` to load the dataset into the corresponding folder like `/data/tinystories`.Since we have already saved the **checkpoint**, if you don't have GPU and you have successfully downloaded the **checkpoint**, you can run `python sample.py` directly.
```bash
python downlaodDataset.py
```
You can modify the code to download the corresponding dataset.
```bash
from datasets import load_dataset
# Load the TinyStories dataset
ds = load_dataset("roneneldan/TinyStories")
# Define the output file path
output_file = "/data/tinytories/TinyStories.txt"
```
Run `prepare.py` to process the text dataset, split it into train and val parts, and save them as binary files.Since the TinyStories.txt dataset is quite large and resources are limited, we only used 472,814 tokens for training and 45,726 tokens for validation. If conditions allow, you can modify the code in `prepare.py` to adjust the training and validation split.
```bash
python ./data/tinystories/prepare.py
```

Then,train the model. To achieve better model training, you need a GPU. If you don't have one, consider using Google's Colab.
```bash
python train.py
```

Now, try to use our model to generate some sentences. If you want to specify a starting word, please add `[--start="any word you want "]` to the command line; otherwise, the default starting word is "Once upona time,".

```bash
python sample.py --out_dir=out [--start="any word you want "]
```

Generates samples like this:
>Once upon a time,house a mole. Tim the an. She decided felt, felt so he was a she. she carefully dug it to snack pulling! From it was a wand was a big and wet! You big crack around it was, there lived it into the boat duck was a great snack!\
Once upon a time there was a little girl. \
One day, Joey's mom said so he was okay for was three raise wake up early raise her have and lowers longer full of the neighbourhood. The voice pe to mix sad that they lost. He smiled. The original knob had heard stories of friends's back

>One day,was in the truck the light faded, the saw looked asked herCan truck. 
The bright yellow butterfly and was so happy to hear how to get out the animals were safe big ones and dad told her called for mom, saw lived a boy called Joey. He decided to everyone."
Once upon a time there was a little girl. The rooster could see that day on it to the ice cream truck.
When he got there was so excitedHello so happy. 
When he loud noise. 
Once upon a time, there was a bad ending for he could could

>Once upon a time,beauty was Lily ran out take the.. Everything thanked an it would. She in her pocket and A were were it
Once upon a time there and wet. A little girl named Grace was looking out of a window. They had an old man outside. He was grumpy and it had an umbrella," the window to they began to get outside. He goodbye and went their separate ways.
Grace felt sorry for the old man, there was a little girl in a big smile. She said had an. She used to say goodbye. He ran into the dust near his paws.



Due to limited resources, we only trained for 2000 iterations, but the results are still fairly reasonable under these conditions. If possible, we recommend training for at least 6000 iterations for better performance.
## Remark
1. This project is implemented based on nanoGPT, and we modified the `model.py` of nanoGPT by referring to the structure of LLaMA, such as choosing **RMSNorm** instead of **LayerNorm** as well as changes to the way datasets are loaded. In `train.py`, code is provided to support both GPU-only environments.
2. The dataset `TinyStories` is too large, so we divided it into smaller subsets, which significantly affected the training results.
3. We initially conducted training on a Chinese dataset and the code is provided to support CPU and GPU. But the results were not very good. If you are interested, you can obtain the relevant source code and files from another repository https://github.com/littlemoon0210/AI_Design_Project.
