# H-Divergence

***Notice: This repo is still under construction.***

### Blob

To replicate the results for alpha = 0.05 from the paper, run
```
python blob_kde.py --exptype power --vtype vmin
```
To replicate the results for alpha = 0.01 from the paper, run
```
python blob_gmm.py --exptype power --vtype vmin
```

### HDGM

To replicate the results from the paper, choose --vtype = vjs, --n = 100,1000,1500,2500, and --d = 3,5,10,15,20 and run
```
python hdgm.py --exptype power --vtype vjs
```

### HIGGS

Get the HIGGS data from here - https://drive.google.com/file/d/1sHIIFCoHbauk6Mkb6e8a_tp1qnvuUOCc/view

To replicate the results from the paper, choose --n = 500,1000,1500,2500,4000,5000 and run
```
python higgs.py
```


### MNIST

Get the fake MNIST data from here - https://drive.google.com/file/d/13JpGbp7PEm4PfZ6VeqpFiy0lHfVpy5Z5/view

To replicate the results from the paper, choose --n = 100,200,300,400,500 and run
```
python mnist.py
```
