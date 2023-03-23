# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

# Use
### you can either use the jupyter notebook or the command line application

## Jupyter Notebook
### open the jupyter notebook and run the cells
### one can easily change the hyperparameters and the model architecture and see the results

## Command Line Application
### train a new network on a data set with train.py
### this supports  densenet121, vgg16 or alexnet as model architecture
### this also supports gpu training
### Basic usage: python train.py data_directory
### Prints out training loss, validation loss, and validation accuracy as the network trains
### Options:
### Set directory to save checkpoints:
```
 python train.py data_dir --save_dir save_directory
```

### Choose architecture:
```
 python train.py data_dir --arch "vgg13" default is densenet121
```

### Set hyperparameters: 
```
python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 (default is 0.5(the 0.54 learning rate is decreasing after every two epochs ) 512 3)
```

### Use GPU for training:
```
 python train.py data_dir --gpu
```

### predict flower name from an image with predict.py along with the probability of that name. That is you'll pass in a single image /path/to/image and return the flower name and class probability.
### Basic usage: 
```
python predict.py /path/to/image checkpoint
```

### Options:
### Return top K most likely classes:
```
 python predict.py input checkpoint --top_k 3
```

### Use a mapping of categories to real names:
```
 python predict.py input checkpoint --category_names cat_to_name.json
```

### Use GPU for inference: 
```
python predict.py input checkpoint --gpu
```

