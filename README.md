[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/obaghirli/syn10-diffusion/main/LICENSE.md)
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)

# SatDM
This repository contains the code and resources for the paper titled  [ Synthesizing Realistic Satellite Image with Semantic Layout Conditioning using Diffusion Models.](https://arxiv.org/pdf/2309.16812.pdf)

<p align="center">
<img src=assets/samples_128.svg />
</p>

## Installation
1. Clone this repository:
```bash
git clone https://github.com/obaghirli/syn10-diffusion.git
cd syn10-diffusion
```
2. Create virtual environment and install the requirements:
```bash
python3.8 -m venv venv
(venv) python -m pip install -r requirements.txt
```
3. Clone the segment-anything repository (to calculate segment-anything score):
```bash
git clone https://github.com/facebookresearch/segment-anything.git
```
4. Download the pretrained segment-anything model:
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

## Data

We trained the model using 0.5-meter resolution data of Baku, which was acquired through the Google Earth platform and manually labeled the acquired data using QGIS software and conducted cross-checks to correct any labeling errors. Dataset description is given below:

<p align="center">
<img src=assets/dataset.png />
</p>


Data can be downloaded from [here](https://drive.google.com/file/d/1jF3LLk9sBrYXgC5iow-5PxE9g6xA2Igm/view?usp=sharing).

You can tile the downloaded data using the following command: 

```bash
python tiler.py \
	--image_root_dir <path_to_images_root_directory> \
	--image_extensions tif \
	--mask_root_dir <path_to_masks_root_directory> \
	--mask_extensions shp \
	--tile_size <tile_size> \
	--tile_overlap <tile_overlap> \
	--tile_keep_ratio <tile_keep_ratio> \
	--downsample_factor <downsample_factor> \
	--image_max_value <maximum_value_of_pixel_of_image> \
	--image_min_value <minimum_value_of_pixel_of_image> \
	--image_channels <number_of_image_channels> \
	--num_train_samples <number_of_train_samples> \
	--num_val_samples <number_of_validation_samples> \
	--save_dir <path_to_save_data>
```

The output directory structure will be as follows:

```
data_dir
├── training
│   ├── images
│   └── annotations
└── validation
    ├── images
    └── annotations
```
## Training

### Training a model from scratch

To train your model, you should first decide on some hyperparameters. The list of hyperparameters and the values we have chosen for experiments can be found in the `configs` folder. After you have created a config file, you can train your model using the following command:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 image_train.py \
	--config <path_to_configs_config.yaml> \
	--data_dir <path_to_dataset> \
	--artifact_dir <path_to_save_outputs> \
	--num_checkpoints <keeps only last n checkpoints, [-1: disable, keep all]>
```
> **Note:** Debugging model (which is a separate mini UnetModel) can be chosen via `--test_model UnetTest`, 
however it is recommended to use the prod model (default model) for the training. If it does not fit into the memory, 
then you can reduce batch size, or depth of the model etc. 

### Resuming training from a checkpoint

You can also resume training from a checkpoint using the following command:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 image_train.py \
	--config <path_to_configs_config.yaml> \
	--data_dir <path_to_dataset> \
	--artifact_dir <path_to_saved_outputs> \
	--num_checkpoints 100 \
	--run_id <id of the job run to continue from e.g.: cec0fba0-4ce5-4fa0-a2b5-f8a0cd36200f> \
	--resume_step 100
```

### Monitoring training progress

To monitor the training progress of your model, you can use the following commands in your terminal or command prompt:

```bash
tensorboard --logdir <path_to_run id e.g.: cec0fba0-4ce5-4fa0-a2b5-f8a0cd36200f/tensorboard>
```
> **Note:** Please check the tensorboard summaries (scalar, histogram, and image) for detailed training information logged at specific intervals mentioned in the config file. For example, eval_freq: 1000 in the config file means that the model will be evaluated every 1000 steps and the results will be logged to tensorboard (image summary).

## Sampling

To sample from the trained model use the following command:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 image_sample.py \
	--config <path_to_config.yaml> \
	--model_path <path_to_model_checkpoint_<iteration number>.pt.tar> \
	--data_dir <path_to_dataset> \
	--artifact_dir <path_to_save_results> \
	--num_batches <sample only this many of batches, ignore for full execution>
```
> **Note:** Add `--test_model` UnetTest argument if the model you are sampling from is a debugging model. Recommended to use the prod model (default model) for the sampling.

> **Note:** Exponential Moving Averages (EMA) of model weights can also be used to sample from.
If EMA checkpoint is available for the iteration of model to sample from, then modify the line as `--model_path /path/to/ema_checkpoint_<iteration number>.pt.tar`. EMA checkpoints can be found in the same directory as the model checkpoints.

## Evaluation

### FID Score

FID score can be calculated using the following command:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 fid.py \
	--real_path <path_to_real_validation_images>  \
	--syn_path <path_to_sampled_validation_images> \
	--batch_size 4 \
	--image_max_value 255 \
	--image_min_value 0 \
	--save_path <path_to_save_results>
```

### Segment-anything Score

Segment-anything score is calculated with the help of [SAM model](https://segment-anything.com/). It is the IOU of the sampled image segmentation mask which is segmented by the help of segment-anything and real mask. For more details, please refer to the paper. 

Segment-anything score can be calculated as follows:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 sam.py \
	--num_points 5 \
	--image_max_value 255 \
	--image_min_value 0 \
	--image_size 64 \
	--batch_size 2 \
	--num_batches <do only this many of batches to collect statistics, ignore for full execution> \
	--num_classes 2 \
	--image_dir <path_to_sampled_images> \
	--annotation_dir <path_to_real_annotations> \
	--sam_checkpoint <path_to_sam_vit_h_4b8939.pth> \
	--model_type vit_h \
	--save_dir <path_to_save_results>

```
> **Note** The above SAM script runs in cpu mode. To run in gpu mode, add `--gpu` argument.

### Similarity Search
 
Similarity search is employed to identify images similar to the generated ones in the training dataset by comparing embedding vectors that are created with pre-trained Inception v3 model, and cosine similarity is utilized to compare these embedding vectors. 

<p align="center">
<img src=assets/similarity_search_128.svg />
</p>

You can run similarity search by using following command: 

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 similarity_search.py \
	--search_path <path_to_real_training_images>  \
	--image_path <path_to_sampled_images_sample_0.npy> <path_to_sampled_images_sample_1.npy> \
	--top_k 3 \
	--batch_size 8 \
	--image_size 64 \
	--image_max_value 255 \
	--image_min_value 0 \
	--save_path <path_to_save_results>
```

### Interpolation
<p align="center">
<img src=assets/interpolate_128.svg />
</p>
To interpolate between two images, use the following command:

```bash
python interpolate.py \
	--image_a_path <path_to_any_images_image_a.npy> \
	--image_b_path <path_to_any_images_image_b.npy> \
	--mask_path <path_to_any_annotations_mask.npy> \
	--model_path <path_to_model_checkpoint_<iteration number>.pt.tar> \
	--config <path_to_configs_config.yaml> \
	--num_steps 500 \
	--lambda_interpolate 0.2 0.8 \
	--save_path <path_to_save_results>
```

### Inpainting
<p align="center">
<img src=assets/inpaint_128.svg />
</p>
You can use the following to use inpainting:

```bash
python inpainting.py \
	--image_path <path_to_any_images_cut_image.npy> \
	--mask_path <path_to_any_annotations_mask_for_cut_image.npy> \
	--model_path <path_to_model_checkpoint_<iteration number>.pt.tar> \
	--config <path_to_configs_config.yaml> \
	--num_steps 500 \
	--save_path <path_to_save_results>
```

### Trajectory
<p align="center">
<img src=assets/trajectory_128.svg />
</p>
We can create the trajectory of image synthesis with the following command:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 trajectory.py \
	--config <path_to_configs_config.yaml> \
	--model_path <path_to_model_checkpoint_<iteration number>.pt.tar> \
	--data_dir <path_to_dataset> \
	--artifact_dir <path_to_save_results> \
	--save_trajectory 1000 900 800 700 600 500 400 300 200 100 0 
```

