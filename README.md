# AdaptiveGaitSegNet: An Innovative Model with Advanced Feature Extraction for Enhanced Parkinson's Disease Gait Recognition

## Project structure

```
AdaptiveGaitSegNet\
├── README.md
├── __pycache__\
├── .idea\
├── gaitist_output\           
│   ├── pre_normal\           
│   │   ├── sub1\ 
│   │   │   ├── GEIs\ 
│   │   │   └── silhouettes\   
│   │   └── ...
│   └── pre_parkinsonian\      
│       ├── sub1\ 
│       │   ├── GEIs\ 
│       │   └── silhouettes\   
│       └── ...
├── modelfile\
│   ├── __pycache__\
│   ├── utils\
│   │   ├── __pycache__\
│   │   ├── __init__.py
│   │   └── evaluator_binary.py
│   │ 
│   ├── __init__.py
│   ├── focal_conv_edge.py
│   ├── data_loader_binary.py
│   ├── data_set_binary.py
│   ├── gaitset_focal_edge.py
│   ├── initialization_binary.py
│   ├── AdaptiveGaitSegNet_binary.py
│   └── sampler.py
├── output\
├── work\
├── .gitignore
├── config_binary.py
├── pretreatment_rotate.py
├── requirements.txt
├── test_binary.py
└── train_binary.py
```

## Prerequisites

- Python 3.8.1
- PyTorch 2.4.1+cu121
- GPU
- Other installation project dependencies:
```bash
   pip install -r requirements.txt
   ```


## Getting started
### Installation

- (Not necessary) Install [Anaconda3](https:\ \ www.anaconda.com\ download\ )
- Install [CUDA 9.0](https:\ \ developer.nvidia.com\ cuda-90-download-archive)
- install [cuDNN7.0](https:\ \ developer.nvidia.com\ cudnn)
- Install [PyTorch](http:\ \ pytorch.org\ )

### Dataset & Preparation
Download [GAIT-IST Dataset](http://www.img.lx.it.pt/GAIT-IST/)
Download [GAIT-IT Dataset](http://www.img.lx.it.pt/GAIT-IT/)

**!!! ATTENTION !!! ATTENTION !!! ATTENTION !!!**

Before training or test, please make sure you have prepared the dataset
by this two steps:
- **Step1:** Organize the directory as: 
`your_dataset_path\gait_type\ subject_ids\ image_type\ views\imge`.
E.g. `gaitist_output\pre_normal\sub1\silhouettes\sub1normal-1_back\000106.png `.
- **Step2:** Cut and align the raw silhouettes with `pretreatment.py`.
(See [pretreatment](#pretreatment) for details.)
Welcome to try different ways of pretreatment but note that
the silhouettes after pretreatment **MUST have a size of 64x64**.

#### Pretreatment

**!!! ATTENTION !!! ATTENTION !!! ATTENTION !!!**

The execution sequence of the preprocessing code should be `video.py → gait_synthesis_visualization.py → pretreatmenr_rotate.py`

`pretreatment_rotate.py` perform geometric alignment and orientation correction on the synthesized silhouette, and unify it to 64×64 to ensure the spatio-temporal consistency of the input model.

Pretreatment your dataset by
```
python pretreatmenr_rotate.py --input_path=".\output\output_synthesis" --output_path=".\output\output_synthesis_rotate" --worker_num=1 --log=TRUE --debug=FALSE
```
- `--input_path` **(NECESSARY)** Root path of raw video.
- `--output_path` **(NECESSARY)** Root path for video output.
- `--log_file` Log file path. #Default: '.\ pretreatment.log'
- `--log` If set as True, the aligned call attempts will be saved. 
Otherwise, only warnings and errors will be saved. #Default: False
- `--worker_num` How many subprocesses to use for data pretreatment. Default: 1
- `--debug` If set as True, all logs will be saved. 
Otherwise, the operation will not be executed. #Default: False

### Configuration 
In `config_binary.py`, you might want to change the following settings:
- `dataset_path` **(NECESSARY)** root path of the dataset 
(for the above example, it is "gaitdata")
- `WORK_PATH` path to save\ load checkpoints
- `CUDA_VISIBLE_DEVICES` indices of GPUs

### Train
Train a model by
```bash
python train_binary.py
```
- `--cache` if set as TRUE all the training data will be loaded at once before the training start.
This will accelerate the training.
**Note that** if this arg is set as FALSE, samples will NOT be kept in the memory
even they have been used in the former iterations. #Default: TRUE

### Evaluation
Evaluate the trained model by
```bash
python test_binary.py
```
- `--iter` iteration of the checkpoint to load. #Default: 10000
- `--batch_size` batch size of the parallel test. #Default: 1
- `--cache` if set as TRUE all the test data will be loaded at once before the transforming start.
This might accelerate the testing. #Default: FALSE

It will output Rank@1 of all three walking conditions. 
Note that the test is **parallelizable**. 
To conduct a faster evaluation, you could use `--batch_size` to change the batch size for test.

## To Do List
- Transformation: The script for transforming a set of silhouettes into a discriminative representation.
