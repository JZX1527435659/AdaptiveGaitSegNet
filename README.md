# AdaptiveGaitSegNet: An Innovative Model with Advanced Feature Extraction for Enhanced  the  Accuracy of Walking Gait recognition.
## Project structure

```
AdaptiveGaitSegNet\
├── README.md
├── __pycache__\
├── .idea\
├── GaitDatasetA-silh\           
│   ├── output\           
│   │   ├── 001\ 
│   │   │   ├── bg-01\
│   │   │   ├── bg-02\
│   │   │   ├── cl-01\
│   │   │   ├── cl-02\
│   │   │   ├── nm-01\
│   │   │   ├── nm-02\
│   │   │   ├── nm-03\
│   │   │   ├── nm-04\
│   │   │   ├── nm-05\
│   │   │   └── nm-06\   
│   │   └── ...
│   ├── pretreatment\      
│   │   ├── 001\ 
│   │   │   ├── bg-01\ 
│   │   │   └── ... 
│   │   └── ...
│   └── pretreatment2\      
│       ├── 001\ 
│       │   ├── bg-01\ 
│       │   └── ...
│       └── ...
├── modelfile\
│   ├── __pycache__\
│   ├── utils\
│   │   ├── __pycache__\
│   │   ├── __init__.py
│   │   ├── basic_blocks.py
│   │   ├── gaitset.py
│   │   └── triplet.py
│   │ 
│   ├── __init__.py
│   ├── basic_blocks.py
│   ├── focal_conv_edge.py
│   ├── data_loader.py
│   ├── data_set.py
│   ├── evaluator.py
│   ├── gaitset.py
│   ├── gaitset_focal_edge.py
│   ├── gaitset_loss.py
│   ├── initialization.py
│   ├── model.py
│   └── sampler.py
├── output\
├── work\
├── demo_checkpoint\
├── .gitignore
├── LICENSE
├── checkpoint_leverage.py
├── config.py
├── gait_synthesis.py
├── pretreatment.py
├── pretreatment.log
├── pretreatment_rotate.py
├── requirements.txt
├── test_ALL.py
├── test_BEST.py
├── test_COPY.py
├── test_FAST.py
├── test_log.txt
├── train.py
└── video.py

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
Download [CASIA](http://www.cbsr.ia.ac.cn/users/szheng/?page_id=71)
Download [OU-MVLP](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html)

**!!! ATTENTION !!! ATTENTION !!! ATTENTION !!!**

Before training or test, please make sure you have prepared the dataset
by this two steps:
- **Step1:** Organize the directory as: 
`your_dataset_path\process_type\subject_id\walking_status\view_angle\imge`.
E.g. `GaitDatasetA-silh\pretreatment\001\bg-01\018\001-bg-01-018-001.png `.
- **Step2:** Cut and align the raw silhouettes with `pretreatment.py`.
(See [pretreatment](#pretreatment) for details.)
Welcome to try different ways of pretreatment but note that
the silhouettes after pretreatment **MUST have a size of 64x64**.

#### Pretreatment
`pretreatment.py` uses the alignment method in
[this paper](https:\ \ ipsjcva.springeropen.com\ articles\ 10.1186\ s41074-018-0039-6).
Pretreatment your dataset by
```
python pretreatment.py --video_root='root_path_of_raw_video' --pretreatment_root='root_path_for_video_output' --synthesis_root='root_path_for_synthesis_output' --final_output_root='root_path_for_final_pretreated_output'
```
- `video_root` **(NECESSARY)** Root path of raw video.
- `pretreatment_root` **(NECESSARY)** Root path for video output.
- `synthesis_root` **(NECESSARY)** Root path for synthesis output.
- `final_output_root` **(NECESSARY)** Root path for final pretreated output.
- `--log_file` Log file path. #Default: '.\ pretreatment.log'
- `--log` If set as True, all logs will be saved. 
Otherwise, only warnings and errors will be saved. #Default: False
- `--worker_num` How many subprocesses to use for data pretreatment. Default: 1

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
