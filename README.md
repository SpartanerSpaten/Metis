Metis
=========

Metis is a light weight neural network framework for Python3.

### setup and installation

```
$ git clone https://github.com/SpartanerSpaten/Metis.git

$ cd Metis

$ python3 setup.py install
```

### Usage

See example.py

```python
from Metis.Metis_CPU_only.Model import Model
from Metis.Metis_CPU_only.Layer import Layer
from Metis.Assets import expit, parse_code
import numpy

mymodel = Model()

mymodel.add(0, Layer(10, 30, expit))
mymodel.add_layer(1, 20, expit)
mymodel.add_layer(2, 4, expit)
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