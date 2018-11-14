import numpy
from Additional._Internal import parse_code, im2col_indices
from numpy.linalg import inv
import math
import warnings

WARNING_LAYER_CANNOT_EXTRACT_FUNCTION_SOURCE_CODE = ResourceWarning(
    "Layer can not extract activation function source code -> no save available")


class Layer:
    def __init__(self, input_size: int, output_size: int, activation_function):
        self.weights = numpy.random.normal(0.0, pow(output_size, -0.5), (output_size, input_size))

        self.input_size = input_size

        self.output_size = output_size

        self.activation_function = activation_function

        self.bias = numpy.random.normal(-0.5, 0.5, (output_size, 1))

        if "__name__" in dir(self.activation_function) and self.activation_function.__name__ == "expit":
            self.function_source = "def expit(x):\nreturn 1 / (1 + pow(math.e, -1 * x))"


        elif "__code__" in dir(self.activation_function):
            self.function_source = parse_code(self.activation_function.__code__.co_filename,
                                              self.activation_function.__code__.co_firstlineno)

        elif self.activation_function.__class__ == numpy.ufunc:
            self.function_source = "def expit(x):\nreturn 1 / (1 + pow(math.e, -1 * x))"

        else:
            Warning(WARNING_LAYER_CANNOT_EXTRACT_FUNCTION_SOURCE_CODE)

            self.function_source = ""

    def forward(self, input_tensor):

        raw_output = numpy.dot(self.weights, input_tensor)
        return self.activation_function(raw_output)

    def backward(self, output):

        raw_input = numpy.dot(self.weights.T, output)
        raw_input -= numpy.min(raw_input)
        raw_input /= numpy.max(raw_input)
        raw_input *= 0.98
        raw_input += 0.01

        return raw_input

    def update_weights(self, error, output, input, learning_rate):

        if input.ndim > 1:
            input = input.flatten().T

        if error.ndim == 1:
            error = error.flatten()
            output = output.flatten()

        temp3 = learning_rate * numpy.dot((error * output * (1.0 - output)),
                                          numpy.array(numpy.transpose(input), ndmin=2))

        self.bias -= learning_rate * 1 * output

        self.weights += temp3

    def calculate_error(self, output_error):
        return numpy.dot(self.weights.T, output_error)

    def get_info(self):
        return ["Layer", self.input_size, self.output_size, self.weights.tolist(), self.function_source]


class Conv2D_Layer(Layer):
    def __init__(self, pattern_n: int, pattern_size: int, activation_function):

        self.activation_function = lambda x: activation_function(numpy.array(x))

        self.pattern_size = pattern_size

        self.pattern_n = pattern_n

        self.filter = numpy.random.normal(0.0, 1, (self.pattern_n, self.pattern_size, self.pattern_size))

    def forward(self, input_tensor):
        cutoffs = im2col_indices(input_tensor, self.pattern_size, self.pattern_size, stride=1, padding=2)
        output = []
        for count, filter in enumerate(self.filter):
            output.append([])
            for cutoff in cutoffs:
                output[count].append(numpy.sum(numpy.dot(filter, cutoff)))

        out = numpy.array(output).reshape((len(output), int(math.sqrt(len(output[0]))), int(math.sqrt(len(output[0])))))
        self.temp = cutoffs
        return self.activation_function(out)

    def calculate_error(self, output_error):

        input_tensor = numpy.array(output_error)

        output = numpy.dot(self.filter.T, input_tensor)

        return output

    def update_weights(self, error, output, input, learning_rate):

        input = im2col_indices(input, self.pattern_size, self.pattern_size, stride=1, padding=2).tolist()

        for count, filter in enumerate(self.filter):
            temp = numpy.zeros(filter.shape)
            temp_output = im2col_indices(output[count], self.pattern_size, self.pattern_size, stride=1,
                                         padding=2).tolist()
            for count_y, y in enumerate(error[count]):

                for count1, value in enumerate(y):
                    temp += numpy.dot(
                        (value * numpy.array(temp_output[count1]) * (1.0 - numpy.array(temp_output[count1]))),
                        numpy.array(numpy.transpose(input[count_y]), ndmin=2))

                self.filter[count] += temp * learning_rate


class Pooling_Layer(Layer):
    def __init__(self, pattern_size: int):

        self.pattern_size = pattern_size

    def forward(self, input_tensor):

        input = im2col_indices(input_tensor, self.pattern_size, self.pattern_size, padding=0,
                               stride=self.pattern_size).tolist()

        output_shape = (int(input_tensor.shape[0] / self.pattern_size), int(input_tensor.shape[1] / self.pattern_size))

        output = numpy.zeros(output_shape).tolist()

        for count, element in enumerate(input):
            temp1 = count - int(count / (self.pattern_size * 2)) * output_shape[0]
            temp2 = int(count / (self.pattern_size * 2))
            if temp2 == output_shape[0] or temp1 == output_shape[1]:
                warnings.warn("Pooling Layer loses data because of not fitting input data")
            else:
                output[temp2][temp1] = max(element)

        return output
