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
mymodel.add(1, Layer(30, 20, expit))
mymodel.add(2, Layer(20, 5, expit))

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
# Creates a new Model
mynewmodel = Model()
# Load the model from mymodel.json
mynewmodel.load_from_file("mymodel.json")

print(mymodel.forward(myinput_vector))
