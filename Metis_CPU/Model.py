import numpy
import Additional._Internal
import Additional.etc
from Metis_CPU.Layer import Layer, Conv2D_Layer
import json


class Model:
    def __init__(self):

        self.layer = []

    def train(self, data: list, learning_rate: float = 0.3, randomise=False, epochs=10):
        """
        Train your nn model to a given trainings data.
        :param data: Your Trainings data set with the shape [[input1,desired1],[input2,desired2]]
        :param learning_rate: learning rate default is 0.3
        :param randomise: should the trainings data be randomised (bool)
        :param epochs: how often the nn should use this trainings data (int)
        """

        for epoch in range(epochs):
            if randomise is True:
                data = Additional.etc.randomise(data)
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

    def forward(self, input_list: list or numpy.ndarray):
        """
        Calculates the output with your given input vector.
        :param input_list: list or ndarray
        :return: ndarray with the size of the last layer
        """
        if type(input_list) is numpy.array:
            output = input_list
        else:

            output = numpy.array(input_list, ndmin=2).T

        for layer in self.layer:
            output = self._prepare_output(output, layer)
            output = layer.forward(output)

        return output

    def backward(self, output: numpy.ndarray or list, function):
        """
        In develop
        :param output:
        :param function:
        :return:
        """

        output = numpy.array(output, ndmin=2).T

        for count in range(len(self.layer)):
            index = len(self.layer) - (count + 1)
            layer = self.layer[index]
            output = function(output)
            output = layer.backward(output)

        return output

    def _prepare_output(self, output, next_layer):

        # if next_layer.__class__ is Layer:
        #    return numpy.array(output.flatten(), ndmin=2).T

        return output

    def add(self, level: int, layer: Layer or Conv2D_Layer):
        """
        Adds another layer to the nn model.
        :param level: integer where the layer should beplaced
        :param layer: Layer object
        """
        if level > len(self.layer) - 1:
            self.layer += [layer]
        else:
            self.layer.insert(level, layer)

    def convert2json(self):
        """
        Converts this nn model to a dict
        :return: nn model dict
        """
        layer_matrix = []
        return_value = {"class": "Metis_Model"}
        for layer in self.layer:
            layer_matrix.append(layer.get_info())
        return_value.update({"layer": layer_matrix})
        return return_value

    def load(self, json: dict):
        """
        Applies the config from the dict to this object
        :param json: config
        :return:
        """
        self.layer = []
        for count, layer in enumerate(json["layer"]):
            func = Additional._Internal.generate_function(layer[4])
            self.add(count, Layer(layer[1], layer[2], func))
            self.layer[count].weights = numpy.array(layer[3])
            self.layer[count].function_source = layer[4]

    def save(self, path: str):
        """
        Saves this model
        :param path: where the saved model should be placed
        :return:
        """

        content = self.convert2json()

        json.dump(content, open(path, "w+"))

    def add_layer(self, layer: int, amount_of_neurons: int, func):
        """
        Waring this function can not be used when there is no layer in this model
        This function generates automaticly a new Layer by a given amount of neurons.
        :param layer: layer > 0 where the new layer should be located
        :param amount_of_neurons: amount of neurons in this layer
        """

        if len(self.layer) > layer:
            layer = len(layer)

        input_size = self.layer[layer - 1].output_size

        self.layer.insert(layer, Layer(input_size, amount_of_neurons, func))

    @staticmethod
    def load_from_file(path: str) -> 'Model':
        """
        Reads the file and generates from it a now Model object
        :param path: path to file
        :return: Model object
        """

        model = Model()

        model.load(json.load(open(path)))

        return model
