# SFDA - Domain Adaptation without Source Data

## Prerequisites
* Ubuntu 18.04    
* Python 3.6+    
* PyTorch 1.5+ (recent version is recommended)     
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
* Install python libraries 
```
conda install -c conda-forge matplotlib
conda install -c anaconda yaml
conda install -c anaconda pyyaml 
conda install -c anaconda scipy
conda install -c anaconda scikit-learn 
conda install -c conda-forge easydict
pip install easydl
```

### Download this repository
* We provide two versions of the repository (with dataset / without dataset) for a flexible experiment

* Full PrDA repository (with dataset): [download link][aa]      
  * In this case, go to ```training and testing``` step directly

[aa]: https://drive.google.com/drive/folders/11g8yOWxIG47G-5vImtX98qrg0Y4UxrGd?usp=sharing

* PrDA repository (without dataset): [download link][a]

[a]: https://drive.google.com/drive/folders/1ndxbQLAkDxxvlPs7E65_6fQ4dNbxXkHR?usp=sharing


*  Visualization of repository structure (Full PrDA repository)

```
|-- APM_update.py
|-- SFDA_test.py
|-- SFDA_train.py
|-- config.py
|-- data.py
|-- lib.py
|-- net.py
|-- office-train-config.yaml
|-- data
|   `-- office
|       |-- domain_adaptation_images
|       |   |-- amazon
|       |   |   `-- images
|       |   |-- dslr
|       |   |   `-- images
|       |   `--- webcam
|       |       `-- images         
|       |-- amazon_31_list.txt
|       |-- dslr_31_list.txt
|       `-- webcam_31_list.txt
|-- pretrained_weights
|   `-- 02
|       `-- domain02accBEST_model_checkpoint.pth.tar
`-- source_pretrained_weights
    `-- 02
        `-- model_checkpoint.pth.tar
```

### Download dataset
* Download the Office31 dataset ([link][b]) and unzip in ```./data/office```     

[b]: https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view

* Download the text file ([link][c]), e.g., amazon_31_list.txt, dslr_31_list.txt, webcam_31_list.txt, in ```./data/office```  

[c]: https://drive.google.com/drive/folders/11wFsBoG--cm7uD0L-7L5X5hprWDCMBpH?usp=sharing


### Download source-pretrained parameters (Fs and Cs of Figure 2 in our main paper)
* Download source-pretrained parameters ([link][d]) in ```./source_pretrained_weights/[scenario_number]```       

[d]: https://drive.google.com/drive/folders/1mkzEl8SHQ0mVFnYV0CvZIdeLstCm2shy?usp=sharing       

  ex) Source-pretrained parameters of A[0] -> W[2] senario should be located in ```./source_pretrained_weights/02```    


## Training and testing

* Arguments required for training and testing are contained in ```office-train-config.yaml  ``` 
* Here is an example of running an experiment on Office31 (default: A -> W)
* Scenario can be changed by editing ```source: 0, target: 2```  in ```office-train-config.yaml```
* We will update the full version of our framework including settings for ```OfficeHome``` and ```Visda-C```  

### Training

*  Run the following command

```
python SFDA_train.py --config office-train-config.yaml
```

### Testing (on pretrained model)

* As a first step, download SFDA pretrained parameters ([link][e]) in ```./pretrained_weights/[scenario_number]```   

   ex) SFDA pretrained parameters of A[0] -> W[2] senario should be located in ```./pretrained_weights/02```    

[e]: https://drive.google.com/drive/folders/1XiWZXsES_oEAI2WMdOBxqjKieA7zOOwZ?usp=sharing

*  or run the training code to obtain pretrained weights

*  Run the following command

```
python SFDA_test.py --config office-train-config.yaml
```



## Experimental results on Office31

* Results using the provided code

|  <center> </center> |  <center>A→W</center> |  <center>D→W</center> |  <center>W→D</center> |  <center>A→D</center> |  <center>D→A</center> |  <center>W→A</center> |  <center>Avg</center> |
|:--------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|Accuracy (%) </center>| <center>91.06 | <center>97.35 | <center>98.99 | <center>91.96 | <center>71.60 | <center>68.62 | <center>86.60 |         
 
 
 
 
 
 
 

