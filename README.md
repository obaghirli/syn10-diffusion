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

### Fake dataset [debugging purposes only]
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
	--save_dir /path/to/save/datset
```

## Training
### Training a model from scratch
```commandline
torchrun --standalone --nnodes=1 --nproc-per-node=1 image_train.py \
	--config /path/to/syn10-diffusion/configs/sat25_test.yaml \
	--data_dir /path/to/dataset \
	--artifact_dir /path/to/save/outputs
```
> **Note:** Debugging model (which is a separate mini UnetModel) can be chosen via `--test_model UnetTest`, 
however it is recommended to use the prod model (default model) for the training. If it does not fit into the memory, 
then you can reduce batch size, or depth of the model etc. 

### Resuming training from a checkpoint
```commandline
torchrun --standalone --nnodes=1 --nproc-per-node=1 image_train.py \
	--config /path/to/syn10-diffusion/configs/sat25_test.yaml \
	--data_dir /path/to/dataset \
	--artifact_dir /path/to/saved/outputs \
	--run_id <id of the job run to continue from e.g.: cec0fba0-4ce5-4fa0-a2b5-f8a0cd36200f> \
	--resume_step 100
```

### Monitoring training progress
```commandline
tensorboard --logdir /path/to/<run id e.g.: cec0fba0-4ce5-4fa0-a2b5-f8a0cd36200f>
```

> **Note:** Please check the tensorboard summaries (scalar, histogram, and image) for detailed training 
> information logged at specific intervals mentioned in the config file. For example, `eval_freq: 1000` in the config file
> means that the model will be evaluated every 1000 steps and the results will be logged to tensorboard (image summary).

## Sampling
### Sampling from a model checkpoint
```commandline
torchrun --standalone --nnodes=1 --nproc-per-node=1 image_sample.py \
	--config /path/to/syn10-diffusion/configs/sat25_test.yaml \
	--model_path /path/to/model_checkpoint_<iteration number>.pt.tar \
	--data_dir /path/to/dataset \
	--artifact_dir /path/to/save/outputs
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
	--save_path /path/to/save/score
```

### Segment-Anything score
```commandline
torchrun --standalone --nnodes=1 --nproc-per-node=1 sam.py \
	--num_points 5 \
	--image_max_value 255 \
	--image_min_value 0 \
	--image_size 64 \
	--batch_size 2 \
	--num_classes 2 \
	--image_dir /path/to/sampled/images \
	--annotation_dir /path/to/real/annotations \
	--sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
	--model_type vit_h \
	--save_dir /path/to/save/score
```

> **Note:** The above SAM script runs in cpu mode. To run in gpu mode, add `--gpu` argument.