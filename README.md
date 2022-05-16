# H-Divergence

***Notice: This repo is still under construction.***

### Blob

```
python blob_kde.py
```
More arguments please check the help.

To replicate the results for alpha = 0.05 from the paper, choose --vtype = vmin or run the previous script directly.

### HDGM

```
python hdgm.py --exptype power --vtype vjs
```
More arguments please check the help.

To replicate the results from the paper, choose --vtype = vjs, --n = 100,1000,1500,2500, and --d = 3,5,10,15,20

### HIGGS

Get the HIGGS data from here - https://drive.google.com/file/d/1sHIIFCoHbauk6Mkb6e8a_tp1qnvuUOCc/view
```
python higgs.py
```
To replicate the results from the paper, choose --n = 500,1000,1500,2500,4000,5000

### MNIST

Get the fake MNIST data from here - https://drive.google.com/file/d/13JpGbp7PEm4PfZ6VeqpFiy0lHfVpy5Z5/view
```
python mnist.py
```
To replicate the results from the paper, choose --n = 100,200,300,400,500
