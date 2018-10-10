import numpy
import Assets
from Layer import Layer, Conv2D_Layer
import json


class Model:
    def __init__(self):

        self.layer = []

    def train(self, data, learning_rate=0.3, randomise=False, epochs=10):

        for epoch in range(epochs):
            if randomise is True:
                data = Assets.randomise(data)
            for element in data:
                self._train_single(element[0], element[1], learning_rate)

    def _train_single(self, input_list, desired_list, learning_rate):
        inputs = numpy.array(input_list, ndmin=2).T
        desireds = numpy.array(desired_list, ndmin=2).T

        output_list = []
        hidden_errors = []

        forwarded_input = inputs

        for layer in self.layer:
            forwarded_input = self._prepare_output(forwarded_input, layer)
            forwarded_input = layer.forward(forwarded_input)
            output_list.append(forwarded_input)

        output_errors = desireds - output_list[-1]
        last = output_errors
        for count in range(0, len(self.layer)):
            index = len(self.layer) - (count + 1)
            layer = self.layer[index]
            output_errors = layer.calculate_error(output_errors)
            hidden_errors.insert(0, output_errors)
        hidden_errors.append(last)

        for count in range(0, len(self.layer)):

            layer = self.layer[count]

            if count >= 1:
                layer.update_weights(hidden_errors[count + 1], output_list[count], output_list[count - 1],
                                     learning_rate)

            else:
                layer.update_weights(hidden_errors[1], output_list[count], inputs, learning_rate)

    def forward(self, input_list):
        if type(input_list) is numpy.array:
            output = input_list
        else:

            output = numpy.array(input_list, ndmin=2).T

        for layer in self.layer:
            output = self._prepare_output(output, layer)
            output = layer.forward(output)

        return output

    def backward(self, output, function):

        output = numpy.array(output, ndmin=2).T

        for count in range(len(self.layer)):
            index = len(self.layer) - (count + 1)
            layer = self.layer[index]
            output = function(output)
            output = layer.backward(output)

        return output

    def _prepare_output(self, output, next_layer):

        return output

    def add(self, level, layer):
        if level > len(self.layer) - 1:
            self.layer += [layer]
        else:
            self.layer.insert(level, layer)

    def convert2json(self):
        layer_matrix = []
        return_value = {"class": "Metis_Model"}
        for layer in self.layer:
            layer_matrix.append(layer.get_info())
        return_value.update({"layer": layer_matrix})
        return return_value

    def load(self, json):
        for count, layer in enumerate(json["layer"]):
            func = Assets.generate_function(layer[4])
            self.add(count, Layer(layer[1], layer[2], func))
            self.layer[count].weights = numpy.array(layer[3])
            self.layer[count].function_source = layer[4]

    def save(self, path):

        content = self.convert2json()

        json.dump(content, open(path, "w+"))

    def load_from_file(self, path):

        content = json.load(open(path))

        self.load(content)
