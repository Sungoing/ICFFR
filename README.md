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

&emsp; Here are the individual TPR and FPR bias comparisons at different global FPRs:

&emsp; ![individual_tpr_bias](https://github.com/Sungoing/ICFFR/blob/main/results/individual_tpr_bias.png)

&emsp; ![individual_fpr_bias](https://github.com/Sungoing/ICFFR/blob/main/results/individual_fpr_bias.png)

## National Bias Evaluation
&emsp; Run the script of eval_national.py, and the result will be saved in ./results directory in a .csv file.

&emsp; Here are the national bias comparisons by IR34 network trained on BalancedFace dataset:

&emsp; ![national_bias](https://github.com/Sungoing/ICFFR/blob/main/results/national_bias.png)

## RFW Bias Evaluation

&emsp; Here are the bias comparisons on RFW by official bias evaluation script.

&emsp; ![rfw_bias](https://github.com/Sungoing/ICFFR/blob/main/results/rfw_bias.png)

## BFW Bias Evaluation

&emsp; Here are the bias comparisons on BFW pairing all the accessible image samples in BFW dataset (Not only the official pairs).

&emsp; ![bfw_bias](https://github.com/Sungoing/ICFFR/blob/main/results/bfw_bias.png)
