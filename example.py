from Metis_CPU_only.Model import Model
from Metis_CPU_only.Layer import Layer
from Assets import expit, parse_code
import numpy

# Generates with numpy random 1D vector
myinput_vector = numpy.random.random(10)
# Generates with numpy random 1D vector
mydesired_output_vector = numpy.random.random(5)

# Init AI Model class
mymodel = Model()
# Adding Layer to the Model
mymodel.add(0, Layer(10, 30, expit))
mymodel.add_layer(1, 20, expit)
mymodel.add_layer(2, 5, expit)

# Prints the forwarded result
print(mymodel.forward(myinput_vector))
print(mydesired_output_vector)

train_data = [[myinput_vector, mydesired_output_vector]]
# Trains the AI model
mymodel.train(train_data)

# Prints the forwarded result
print(mymodel.forward(myinput_vector))

# Save the file to mymodel.json
mymodel.save("mymodel.json")
# Creates a new Model and loads the data from the mymodel.json file
mynewmodel = Model.load_from_file("mymodel.json")


print(mymodel.forward(myinput_vector))
