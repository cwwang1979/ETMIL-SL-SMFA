# ETMIL-SL-SMFA
Ensemble Transformer-based Multiple Instance Learning with Soft Loss Training and Soft Multiclass Fusion Augmentation (ETMIL-SL-SMFA)

## Associated Publications

Wang et al. (In submission) Deep Learning to Identify Tumor Origins Using Cytology or Cell Block Whole Slide Images

## Setup

#### Requirerements
- Ubuntu 18.04
- GPU Memory => 12 GB
- GPU driver version >= 470.182.03
- GPU CUDA >= 11.4
- Python (3.7.11), h5py (2.10.0), opencv-python (4.2.0.34), PyTorch (1.10.1), torchvision (0.11.2), pytorch-lightning (1.2.3).

#### Download
Execution file, configuration file, and models are download from the [zip](https://drive.google.com/file/d/1jQRKKcIbgVhmQj-Pj_LB-rk1V4rrbC9F/view?usp=drive_link) file.  (Please see the code availability section in the manuscript for the password to decompress the file.)

## Steps
#### 1. Installation

Please refer to the following instructions.
```
# create and activate the conda environment
conda create -n tmil python=3.7 -y
conda activate tmil

# install pytorch
## pip install
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
## conda install
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

# install related package
pip install -r requirements.txt
```

#### 1. Tissue Segmentation and Patching

Place the whole slide image in ./DATA
```
./DATA/XXXX
├── slide_1.svs
├── slide_2.svs
│        ⋮
└── slide_z.svs
  
```

Then, in a terminal, run:
```
python create_patches.py --source DATA/XXXX --save_dir DATA_PATCHES/XXXX --patch_size 256 --preset tcga.csv --seg --patch --stitch

```

After running in a terminal, the result will be produced in the folder named 'DATA_PATCHES/XXXX', which includes the masks and the stitches in .jpg, and the coordinates of the patches will be stored in HD5F files (.h5) like the following structure.
```
DATA_PATCHES/XXXX/
├── masks/
│   ├── slide_1.jpg
│   ├── slide_2.jpg
│   │       ⋮
│   └── slide_z.jpg
│
├── patches/
│   ├── slide_1.h5
│   ├── slide_2.h5
│   │       ⋮
│   └── slide_z.h5
│
├── stitches/
│   ├── slide_1.jpg
│   ├── slide_2.jpg
│   │       ⋮
│   └── slide_z.jpg
│
└── process_list_autogen.csv
```


#### 2. Feature Extraction

In the terminal, run:
```
CUDA_VISIBLE_DEVICES=0,1 python extract_features.py --data_h5_dir DATA_PATCHES/XXXX/ --data_slide_dir DATA/XXXX --csv_path DATA_PATCHES/XXXX/process_list_autogen.csv --feat_dir DATA_FEATURES/XXXX/ --batch_size 512 --slide_ext .svs

```

example features results:
```
DATA_FEATURES/XXXX/
├── h5_files/
│   ├── slide_1.h5
│   ├── slide_2.h5
│   │       ⋮
│   └── slide_z.h5
│
└── pt_files/
    ├── slide_1.pt
    ├── slide_2.pt
    │       ⋮
    └── slide_z.pt
```

#### 3. Training and Testing List
Prepare the training, validation, and testing list containing the labels of the files and put it into the ./dataset_csv folder. (The CSV sample "fold0.csv" is provided)

Example of the CSV files:
|      | train          | train_label     | val        | val_label | test        | test_label |  
| :--- | :---           |  :---           | :---:      |:---:      | :---:      |:---:      | 
|  0   | train_slide_1        | 1               | val_slide_1    |   0       | test_slide_1    |   0       | 
|  1   | train_slide_2        | 0               | val_slide_2    |   1       | test_slide_2    |   0       |
|  ... | ...            | ...             | ...        | ...       | ...        | ...       |
|  z-1   | train_slide_z        | 1               |     |          |    |          |



#### 4. Inference 

To generate the prediction outcome of the ETMIL model, containing E base models:
```
python ensemble_inf_multiple.py --stage='test' --config='Config/TMIL.yaml'  --gpus=0 --top_fold=E
```
On the other hand, to generate the prediction outcome of the TMIL model, containing a single base model:
```
python ensemble_inf_multiple.py --stage='test' --config='Config/TMIL.yaml'  --gpus=0 --top_fold=1
```

To set up the ETMIL model for different tasks: 
1. Open the Config file ./Config/TMIL.yaml
2. Change the log_path in Config/TMIL.yaml to the corresponding model path
   
(e.g., For identifying the primary origin of malignant cells in pleural and ascitic fluids directly from WSIs of cytological smears: please set the parameter "log_path" in Config/TMIL.yaml as "./log/Cytology/ETMIL-SL-SMFA/")

The model of each task has been stored in the zip file with the following file structure: 
```
log/
├── Cytology/
│   └── ETMIL-SL-SMFA
└── CellBlock/
    └── ETMIL-SL-SMFA
```


## Training
#### Preparing Training Splits

To create an E-fold for training and validation sets from the training list. 
```
dataset_csv/
├── fold0.csv
├── fold1.csv
│       ⋮
└── foldE.csv
```

#### Training

Run this code in the terminal to train E-fold:
```
for((FOLD=0;FOLD<N;FOLD++)); do python train.py --stage='train' --config='Config/TMIL.yaml' --gpus=0 --fold $FOLD ; done
```

Run this code in the terminal to train one single fold:
```
python train.py --stage='train' --config='Config/TMIL.yaml' --gpus=0 --fold=0
```


## License
This Python source code is released under a Creative Commons license, which allows for personal and research use only. For a commercial license please contact Prof Ching-Wei Wang. You can view a license summary here:  
http://creativecommons.org/licenses/by-nc/4.0/


## Contact
Prof. Ching-Wei Wang  
  
cweiwang@mail.ntust.edu.tw; cwwang1979@gmail.com  
  
National Taiwan University of Science and Technology

