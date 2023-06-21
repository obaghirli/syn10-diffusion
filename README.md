[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/obaghirli/syn10-diffusion/main/LICENSE.md)
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)

# Synthesizing Photorealistic Satellite Imagery with Semantic Layout Conditioning using Denoising Diffusion Probabilistic Models




## Development Environment

|       | Version           |
|-------|------------------:|
| OS    | Ubuntu 22.04.2 LTS|
| Python|           3.8.16 |
| PyTorch|           2.0.0 |
| CUDA  |           11.7   |
| cuDNN |           8.5.0  |
| GPU   | GeForce RTX 2060 Mobile |
| Driver|     515.65.01    |


## Installation
1. Clone the syn10-diffusion repository:
```bash
git clone https://github.com/obaghirli/syn10-diffusion.git
```
2. Create virtual environment and install the requirements:
```bash
python3.8 -m venv venv
(venv) python -m pip install -r requirements.txt
```
3. Clone the segment-anything repository:
```bash
git clone https://github.com/facebookresearch/segment-anything.git
```
4. Download the pretrained model:
```bash
wget -O sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

```
5. Set system paths:  
Create a file named `syn10_diffusion.pth` in the `venv/lib/python3.8/site-packages` directory and add the following lines:
```
/path/to/syn10-diffusion
/path/to/syn10-diffusion/syn10_diffusion
/path/to/syn10-diffusion/scripts
/path/to/segment-anything
/path/to/segment-anything/segment_anything
```

## Dataset creation

### Dummy dataset [debugging purposes only]
```commandline
python faker.py \
	--image_size 64 \
	--image_channels 3 \
	--image_max_value 255 \
	--image_min_value 0 \
	--num_train_samples 1000 \
	--num_val_samples 100 \
	--save_dir /path/to/save/dataset
```

### Real dataset
Link to dataset: https://drive.google.com/file/d/1jF3LLk9sBrYXgC5iow-5PxE9g6xA2Igm/view?usp=sharing
```commandline
python tiler.py \
	--image_root_dir /path/to/images/root/directory \
	--image_extensions tif \
	--mask_root_dir /path/to/masks/root/directory \
	--mask_extensions shp \
	--tile_size 128 \
	--tile_overlap 0.0 \
	--tile_keep_ratio 0.1 \
	--downsample_factor 2 \
	--image_max_value 255 \
	--image_min_value 0 \
	--image_channels 3 \
	--num_train_samples 1000 \
	--num_val_samples 100 \
	--save_dir /path/to/save/data
```

## Training
### Training a model from scratch
```commandline
torchrun --standalone --nnodes=1 --nproc-per-node=1 image_train.py \
	--config /path/to/configs/config.yaml \
	--data_dir /path/to/dataset \
	--artifact_dir /path/to/save/outputs \
	--num_checkpoints <keeps only last n checkpoints, [-1: disable, keep all]>
```
> **Note:** Debugging model (which is a separate mini UnetModel) can be chosen via `--test_model UnetTest`, 
however it is recommended to use the prod model (default model) for the training. If it does not fit into the memory, 
then you can reduce batch size, or depth of the model etc. 

### Resuming training from a checkpoint
```commandline
torchrun --standalone --nnodes=1 --nproc-per-node=1 image_train.py \
	--config /path/to/configs/config.yaml \
	--data_dir /path/to/dataset \
	--artifact_dir /path/to/saved/outputs \
	--num_checkpoints 100 \
	--run_id <id of the job run to continue from e.g.: cec0fba0-4ce5-4fa0-a2b5-f8a0cd36200f> \
	--resume_step 100
```

### Monitoring training progress
```commandline
tensorboard --logdir /path/to/<run id e.g.: cec0fba0-4ce5-4fa0-a2b5-f8a0cd36200f/tensorboard>
```

> **Note:** Please check the tensorboard summaries (scalar, histogram, and image) for detailed training 
> information logged at specific intervals mentioned in the config file. For example, `eval_freq: 1000` in the config file
> means that the model will be evaluated every 1000 steps and the results will be logged to tensorboard (image summary).

## Sampling
### Sampling from a model checkpoint
```commandline
torchrun --standalone --nnodes=1 --nproc-per-node=1 image_sample.py \
	--config /path/to/configs/config.yaml \
	--model_path /path/to/model_checkpoint_<iteration number>.pt.tar \
	--data_dir /path/to/dataset \
	--artifact_dir /path/to/save/results \
	--num_batches <sample only this many of batches, ignore for full execution>
```

> **Note:**  Add `--test_model UnetTest` argument if the model you are sampling from is a debugging model.
> Recommended to use the prod model (default model) for the sampling. 

> **Note:** Exponential Moving Averages (EMA) of model weights can also be used to sample from.  
> If EMA checkpoint is available for the iteration of model to sample from, then modify the line as 
> `--model_path /path/to/ema_checkpoint_<iteration number>.pt.tar`. EMA checkpoints can be found in the
> same directory as the model checkpoints.

## Evaluation
### FID score
```commandline
torchrun --standalone --nnodes=1 --nproc-per-node=1 fid.py \
	--real_path /path/to/real/validation/images  \
	--syn_path /path/to/sampled/validation/images \
	--batch_size 4 \
	--image_max_value 255 \
	--image_min_value 0 \
	--save_path /path/to/save/results
```

### Segment-Anything score
```commandline
torchrun --standalone --nnodes=1 --nproc-per-node=1 sam.py \
	--num_points 5 \
	--image_max_value 255 \
	--image_min_value 0 \
	--image_size 64 \
	--batch_size 2 \
	--num_batches <do only this many of batches to collect statistics, ignore for full execution> \
	--num_classes 2 \
	--image_dir /path/to/sampled/images \
	--annotation_dir /path/to/real/annotations \
	--sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
	--model_type vit_h \
	--save_dir /path/to/save/results
```

> **Note:** The above SAM script runs in cpu mode. To run in gpu mode, add `--gpu` argument.

### Similarity search
```commandline
torchrun --standalone --nnodes=1 --nproc-per-node=1 similarity_search.py \
	--search_path /path/to/real/training/images  \
	--image_path /path/to/sampled/images/sample_0.npy /path/to/sampled/images/sample_1.npy \
	--top_k 3 \
	--batch_size 8 \
	--image_size 64 \
	--image_max_value 255 \
	--image_min_value 0 \
	--save_path /path/to/save/results
```

### Interpolation
```commandline
python interpolate.py \
	--image_a_path /path/to/any/images/image_a.npy \
	--image_b_path /path/to/any/images/image_b.npy \
	--mask_path /path/to/any/annotations/mask.npy \
	--model_path /path/to/model_checkpoint_<iteration number>.pt.tar \
	--config /path/to/configs/config.yaml \
	--num_steps 500 \
	--lambda_interpolate 0.2 0.8 \
	--save_path /path/to/save/results
```


### Inpainting
```commandline
python inpainting.py \
	--image_path /path/to/any/images/cut_image.npy \
	--mask_path /path/to/any/annotations/mask_for_cut_image.npy \
	--model_path /path/to/model_checkpoint_<iteration number>.pt.tar \
	--config /path/to/configs/config.yaml \
	--num_steps 500 \
	--save_path /path/to/save/results
```

### Experimentation

Directory structure
```
$HOME
├── syn10-diffusion (repo)
├── segment-anything (repo)
├── sam_vit_h_4b8939.pth (SAM checkpoint)
├── raw_data (iac dataset)
├── data (output of tiler.py)
├── results (artifact_dir)
```

Fixed parameters

Image
- image_channels: 3
- image_max_value: 255
- image_min_value: 0
- num_classes: 2

Diffusion
- s: 0.008
- max_beta: 0.999
- lambda_variational: 0.001
- num_diffusion_timesteps: 1000

Model
- p_uncond: 0.2
- lr: 0.00002
- grad_clip: 1.0
- weight_decay: 0.05
- norm_channels: 32
- model_input_channels: 3
- model_output_channels: 6
- ema_decay: 0.9999
- guidance: 1.5
- attn_resolutions: [32, 16, 8]
- num_resnet_blocks: 2
- t_embed_mult: 4.0
- y_embed_mult: 1.0
- lr_scheduler_t_mult: 1


Checkpoint
- ema_delay: 10000

Experiment 1 (64x64, 16gb 1080Ti)
- image_size: 64
- dropout: 0.1
- num_epochs: 257 (num_steps: 400k)
- channel_mult: [1, 2, 3, 4]
- model_channels: 64
- head_channels: 32
- lr_scheduler_t_0: 257
- train_batch_size: 32
- sample_batch_size: 32
- checkpoint_freq: 20000
- tensorboard_freq: 400
- eval_freq: 20000

Experiment 2 (128x128, 32gb Tesla v100)
- image_size: 128
- dropout: 0.0
- num_epochs: 801 (num_steps: 2.5m)
- channel_mult: [1, 1, 2, 3, 4]
- model_channels: 128
- head_channels: 64
- lr_scheduler_t_0: 801
- train_batch_size: 16
- sample_batch_size: 16
- checkpoint_freq: 50000
- tensorboard_freq: 2500
- eval_freq: 50000

Evaluation for each experiment
- fid (~10 models)
- sam (best model)
- similarity search (best model)
- interpolation (best model)
- inpainting (best model)

Human evaluation
- Preference between synthetic vs real images

Notes
- NVIDIA Tesla V100
- Dataset complexity analysis
- Batch size per GPU / Throughput images per V100-sec
- Forward pass FLOPs

Experiment 1 scripts
```commandline
python tiler.py \
	--image_root_dir /home/orkhan/iac \
	--image_extensions tif \
	--mask_root_dir /home/orkhan/iac \
	--mask_extensions shp \
	--tile_size 128 \
	--tile_overlap 0.5 \
	--tile_keep_ratio 0.01 \
	--downsample_factor 2 \
	--image_max_value 255 \
	--image_min_value 0 \
	--image_channels 3 \
	--num_train_samples 50000 \
	--num_val_samples 5000 \
	--save_dir /home/orkhan/data

torchrun --standalone --nnodes=1 --nproc-per-node=1 image_train.py \
	--config /home/orkhan/syn10-diffusion/configs/experiment_1_64.yaml \
	--data_dir /home/orkhan/data \
	--artifact_dir /home/orkhan/results \
	--num_checkpoints -1

torchrun --standalone --nnodes=1 --nproc-per-node=1 image_sample.py \
	--config /home/orkhan/syn10-diffusion/configs/experiment_1_64.yaml \
	--model_path /home/orkhan/results/438143b4-d47b-4d20-891a-3c37ef753848/checkpoint/ema_checkpoint_40000.pt.tar \
	--data_dir /home/orkhan/data \
	--artifact_dir /home/orkhan/results
	
torchrun --standalone --nnodes=1 --nproc-per-node=1 fid.py \
	--real_path /home/orkhan/data/validation/images  \
	--syn_path /home/orkhan/results/sampler_ema_checkpoint_40000_f816d1e3-a44d-43a4-b45d-f4fe7738d22c/images \
	--batch_size 8 \
	--image_max_value 255 \
	--image_min_value 0 \
	--save_path /home/orkhan/results
	
torchrun --standalone --nnodes=1 --nproc-per-node=1 sam.py \
	--num_points 5 \
	--image_max_value 255 \
	--image_min_value 0 \
	--image_size 64 \
	--batch_size 2 \
	--num_classes 2 \
	--image_dir /home/orkhan/results/sampler_ema_checkpoint_40000_f816d1e3-a44d-43a4-b45d-f4fe7738d22c/images \
	--annotation_dir /home/orkhan/results/sampler_ema_checkpoint_40000_f816d1e3-a44d-43a4-b45d-f4fe7738d22c/annotations \
	--sam_checkpoint /home/orkhan/sam_vit_h_4b8939.pth \
	--model_type vit_h \
	--save_dir /home/orkhan/results
	
torchrun --standalone --nnodes=1 --nproc-per-node=1 similarity_search.py \
	--search_path /home/orkhan/data/training/images  \
	--image_path /home/orkhan/results/sampler_ema_checkpoint_40000_f816d1e3-a44d-43a4-b45d-f4fe7738d22c/images/7316c8b468964f34969b17d412d62e9d.npy \
	--top_k 3 \
	--batch_size 8 \
	--image_size 64 \
	--image_max_value 255 \
	--image_min_value 0 \
	--save_path /home/orkhan/results

python interpolate.py \
	--image_a_path /home/orkhan/results/sampler_ema_checkpoint_40000_f816d1e3-a44d-43a4-b45d-f4fe7738d22c/images/7316c8b468964f34969b17d412d62e9d.npy \
	--image_b_path /home/orkhan/results/sampler_ema_checkpoint_40000_f816d1e3-a44d-43a4-b45d-f4fe7738d22c/images/3948e16ce9624364ab086f3f79b32189.npy \
	--mask_path /home/orkhan/results/sampler_ema_checkpoint_40000_f816d1e3-a44d-43a4-b45d-f4fe7738d22c/annotations/7316c8b468964f34969b17d412d62e9d.npy \
	--model_path /home/orkhan/results/438143b4-d47b-4d20-891a-3c37ef753848/checkpoint/ema_checkpoint_40000.pt.tar \
	--config /home/orkhan/syn10-diffusion/configs/experiment_1_64.yaml \
	--num_steps 500 \
	--lambda_interpolate 0.2 0.8 \
	--save_path /home/orkhan/results

python inpainting.py \
	--image_path /home/orkhan/results/sampler_ema_checkpoint_40000_f816d1e3-a44d-43a4-b45d-f4fe7738d22c/images/7316c8b468964f34969b17d412d62e9d.npy \
	--mask_path /home/orkhan/results/sampler_ema_checkpoint_40000_f816d1e3-a44d-43a4-b45d-f4fe7738d22c/annotations/7316c8b468964f34969b17d412d62e9d.npy \
	--model_path /home/orkhan/results/438143b4-d47b-4d20-891a-3c37ef753848/checkpoint/ema_checkpoint_40000.pt.tar \
	--config /home/orkhan/syn10-diffusion/configs/experiment_1_64.yaml \
	--num_steps 500 \
	--save_path /home/orkhan/results
```

Experiment 2 scripts
```commandline
python tiler.py \
	--image_root_dir /home/orkhan/iac \
	--image_extensions tif \
	--mask_root_dir /home/orkhan/iac \
	--mask_extensions shp \
	--tile_size 128 \
	--tile_overlap 0.5 \
	--tile_keep_ratio 0.01 \
	--downsample_factor 1 \
	--image_max_value 255 \
	--image_min_value 0 \
	--image_channels 3 \
	--num_train_samples 50000 \
	--num_val_samples 5000 \
	--save_dir /home/orkhan/data
```