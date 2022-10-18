# ICFFR
&emsp; This project is for the fair face recognition method IC-FFR and the NFW dataset for individual and national bias evaluation.

## Train
&emsp; Part of the codes are referenced to [TFace Projects](https://github.com/Tencent/TFace). The training data can be prepared in the format of tfrecord following that project instructions. After that, you can start the training by:

&emsp; cd icffr/

&emsp; bash local_train.sh

&emsp; The training configurations can be modified according to your environment in configs/config_icffr.yaml.


## Test

## Data Preparation
&emsp; Download the images of our NFW from [GoogleDrive](https://drive.google.com/drive/folders/13bMbZEUcap0yNPJo57I1clKptwyD6ELK?usp=sharing), and put the whole images directory into ./data/

&emsp; cd ./data/

&emsp; unzip national_pospairs.zip

&emsp; unzip national_negpairs.zip

&emsp; cd images/

&emsp; unzip images.zip

## Model Preparation
&emsp; Download the pretrained model of our IC-FFR from [GoogleDrive](https://drive.google.com/drive/folders/1C-Jz0eYm4bwpPhP-EzQXazIwnNZvIf7e?usp=sharing), and put all the .pth files into ./model

## Individual Bias Evaluation
&emsp; Run the script of eval_individual.py, and the result will be saved in ./results directory in a .csv file.

&emsp; Example of the individual bias:

&emsp; ![balance_ir34_individual_tpr](https://github.com/Sungoing/ICFFR/blob/main/results/ba_34_individual_tpr.png)

## National Bias Evaluation
&emsp; Run the script of eval_national.py, and the result will be saved in ./results directory in a .csv file.

&emsp; Example of the national bias:

&emsp; ![balance_ir34_national](https://github.com/Sungoing/ICFFR/blob/main/results/ba_34_national.png)

