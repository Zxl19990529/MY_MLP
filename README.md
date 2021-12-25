# Image Synthesis Technology, Autumn 2021

Homework 1 for Image Synthesis Technology, Autumn 2021

张心亮，2021244024

## Environment

```py
python=3.6
matplotlib
numpy
```

Anaconda3 base environment is recommanded. Just the base environment is enough.

## Usage

### Network

The basic layer like Linear layer and activate function layer are defined in ```basic_layer.py```. The network is predifined in ```models.py```. Here is an example of defining a fully connected network in your code.

```py
model = Network()
model.add(Linear('fc1', 1, 256, 0.001))
model.add(Tanh('tanh1'))
model.add(Linear('fc2', 256, 256, 0.001))
model.add(Tanh('tanh2'))
model.add(Linear('fc3', 256, 128, 0.001))
model.add(Tanh('tanh3'))
model.add(Linear('fc4',128,1,0.001))
model.train()
```

All the configures of the network and training strategy is defined in ```config.yml```

### Task 1 
Firstly, generate the dataset. Run the command: ```python generate_data.py``` and the file "a1b2c2d2.csv" will be generated in the current folder.

Then run  ```python mydnn.py --log_dir 2_layer_log_tanh --data a1b2c2d2.csv```, and the training and testing results of FCN-S-t will be saved in folder ```2_layer_log_tanh```

The training and testing code is ```mydnn.py```, the network is set to FCN-S-t in default.

### Task2
Firstly prepare the MNIST dataset in current folder. The folder is recommanded to organised as :

```
- MNIST
    ├─t10k-images.idx3-ubyte
    ├─t10k-labels.idx1-ubyte
    ├─train-images.idx3-ubyte
    └─train-labels.idx1-ubyte
```

The training and testing code is ```my_mnist.py```. Run directly ```python my_mnist.py --data MNIST --log_dir ./mnist_log/3_layer_tanh_sigmoid```. in default, the network is FCN-B-t.

Then the training and testing results of FCN-B-t on Task 2 is generated in ```mnist_log\3_layer_tanh_sigmoid```

### Visualization for Task 1
<table>
    <tr>
        <td><center><img src="gifs\fcn_s_r_1.gif" width="210"/><br>FCN-S-r</center></td>
        <td><center><img src="gifs\fcn_b_r_1.gif" width="210"/><br>FCN-B-r</center></td>
        <td><center><img src="gifs\fcn_l_r_1.gif" width="210"/><br>FCN-L-r</center></td>
    </tr>
</table>

<table>
    <tr>
        <td><center><img src="gifs\fcn_s_t_1.gif" width="210"/><br>FCN-S-t</center></td>
        <td><center><img src="gifs\fcn_b_t_1.gif" width="210"/><br>FCN-B-t</center></td>
        <td><center><img src="gifs\fcn_l_t_1.gif" width="210"/><br>FCN-L-t</center></td>
    </tr>
</table>