# MSTA-VAE: Enhancing Trend Representation for Time Series Forecasting via Multi-Scale and Time-Aware Fusion![image](https://github.com/allen30404/LaST_Paper/assets/61857422/5c9c6a71-1787-45bc-abb4-1d9552188bb0)

## Dataset

We conducted extensive experiments on seven real-world benchmark datasets from four covering the categories of mainstream time series forecasting applications.  

Please download from the following buttons and place them into `datasets` folder.

[![](https://img.shields.io/badge/Download-Dataset-%234285F4?logo=GoogleDrive&labelColor=lightgrey)](https://drive.google.com/drive/folders/13Ae_qDDxTQDroHCKUIG4xp3Sfi6yuhjX?usp=sharing)



## Usage

#### Requirements

# import virtual environment
conda env create -f MSTA_VAE.yaml

# activate environment
conda activate MSTA_VAE



#### Run code

To train and evaluate LaST framework on a dataset, run the following command:

```shell
python run.py --data <dataset_name>  --features <forecasting_mode>  --seq_len <input_length>  --pred_len <pred_length>  --latent_size <latent_size>  --batch_size <batch_size>  --patience <patience>  --seed <random_seed>
```

The detailed descriptions about the arguments are as following:

| Parameter name      | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| dataset_name        | The dataset name can be selected from ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange_rate", "Electricity", "Weather"] |
| forecasting_mode    | A value in ["S", "M"]. "S" denotes univariate forecasting while "M" denotes multivariate forecasting. |
| input_length        | The input (historical) sequence length, default is 201.      |
| pred_length         | The output (forecasting) sequence length.                    |
| latent_size         | The dimension of latent representations, default is 128.     |
| batch_size          | Batch size, default is 32.                                   |
| patience            | The steps of early stop strategy in training.                |
| random_seed         | The random seed.                                             |
| Encoder_Muti_Scale  | Use Multi_Scale in Encoder or not.                           |
| Decoder_Muti_Scale  | Use Multi_Scale in DEcoder or not.                           |
| Encoder_Fusion      | Use Fusion in Encoder or not.                                |
| Decoder_Fusion      | Use Fusion in Decoder or not.                                |
| Mean_Var_Model      | Mean_Var_Model[mlp,Conv], default is mlp                     |



## Directory Structure

The code directory structure is shown as follows:
```shell
LaST
├── datasets  # seven datasets files
│   ├── ETTh1.csv
│   ├── electricity.csv
│   └── weather.csv
├── expriments  # training, validation, and test code of MSTA-VAE
│   ├── exp_basic.py
│   └── exp_LaST.py
├── models  # code of LaST and its dependencies
│   ├── LaST.py  # MSTA-VAE main code
│   └── utils.py  # modules for LaST including autocorrelation, cort, etc.
├── utlis
│   ├── data_loader.py  # data loading and preprocessing code
│   ├── metrics.py  # metrics for evaluation
│   ├── timefeatures.py  # extract time-related features
│   └── tools.py  # tools for training, such as early stopping and learning rate controls 
├── LICENSE  # code license
├── run.py  # entry for model training, validation, and test 
└── README.md  # This file
```
