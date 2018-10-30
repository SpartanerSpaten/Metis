Metis
=========

Metis is a light weight neural network framework for Python3.

### setup and installation

```
$ git clone https://github.com/SpartanerSpaten/Metis.git

$ cd Metis

$ python3 setup.py install
```

Perhaps you have to execute this commands as root user.

If you want Cuda support add --pycuda to the command from above.
```
$ python3 setup.py install --pycuda
```

There is another option when you only want the pycuda version without the cpu one
add --only_pycuda.

### Usage

See example.py

```python
from Metis_CPU.Model import Model
from Metis_CPU.Layer import Layer
import Additional.Functions as f
import numpy

mymodel = Model()

mymodel.add(0, Layer(10, 30, f.expit))
mymodel.add_layer(1, 20, f.expit)
mymodel.add_layer(2, 4, f.expit)
```
This sniped of code configurated our NN Model in all layers is used the expit function
furthermore this model have 10 input neurons 20 hidden and 4 output neurons.

```python
train_data = [[myinput_vector, mydesired_output_vector]]

mymodel.train(train_data)
```

This Trains your NN model the your previusly defined traindata.

```python
mymodel.save("mymodel.json")

mynewmodel = Model.load_from_file("mymodel.json")
```

This saves the "old" NN model in the file 'mymodel.json' and loads a new model from this file.