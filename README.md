# SFDA-Domain-Adaptation-without-Source-Data
This is an anonymous GitHub for a NeurIPS2020 double-blind submission     
Note: We will refer the related GitHub codes after public release for keeping anonymity

## Prerequisites
* Ubuntu 18.04    
* Python 3.6+    
* PyTorch 1.3+ (recent version is recommended)     
* NVIDIA GPU (>= 12GB)      
* CUDA 10.0 (optional)         
* CUDNN 7.5 (optional)         

## Getting Started

### Installation
* Configure virtual (anaconda) environment
```
conda create -n env_name python=3.6
source activate env_name
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```
* Install python libraries (!!!!!!!!!!!!!!!! change)
```
conda install -c conda-forge matplotlib
conda install -c anaconda yaml
conda install -c anaconda pyyaml 
conda install -c anaconda scipy
conda install -c anaconda pandas 
conda install -c anaconda scikit-learn 
conda install -c conda-forge opencv
conda install -c anaconda seaborn
conda install -c conda-forge tqdm
```

### Download this repository
* Download 
Link: [SFDA repository][a]

[a]: https://git@github.com:youngryan1993/SFDA-Domain-Adaptation-without-Source-Data.git


### Download Dataset
* Download the Office31 dataset ([link][b]) and unzip in ```../data/office```     

[b]: https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view

* Download the text file (amazon_31_list.txt, dslr_31_list.txt, webcam_31_list.txt / [link][c]) to ```../data/office```  

[c]: https://drive.google.com/drive/folders/11wFsBoG--cm7uD0L-7L5X5hprWDCMBpH?usp=sharing


### Download source-pretrained parameters (Fs and Cs in the main paper Figure2)
* Download source-pretrained parameters ([link][d]) in ```./modelsave_official/office/[scenario_number]```    

[d]: https://drive.google.com/drive/folders/1mkzEl8SHQ0mVFnYV0CvZIdeLstCm2shy?usp=sharing

   ex) Source-pretrained parameters of A[0] -> W[2] senario should be located in ```./modelsave_official/office/02```    


## Training and testing

* Arguments required for training is contained in ```office-train-config.yaml  ``` 
* Here is a simple example of running an experiment on only one scenario in Office31 (A -> W)     
* We will update our full version after a rebuttal period


## Training
```
python SFDA_train.py --config office-train-config.yaml
```

## Testing on pretrained model
```
python SFDA_test.py --config office-train-config.yaml
```
