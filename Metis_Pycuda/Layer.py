WARNING_LAYER_CANNOT_EXTRACT_FUNCTION_SOURCE_CODE = ResourceWarning(
    "Layer can not extract activation function source code -> no save available")
ERROR_PYCUDA_MISSING = ImportError("PYCUDA IS MISSING")
ERROR_NUMPY_MISSING = ImportError("NUMPY IS MISSING")
ERROR_SCIKIT_CUDA_MISSING = ImportError("SCIKIT_CUDA IS MISSING")

import numpy
import pycuda.gpuarray as gpuarray
import skcuda.linalg as linalg
import numpy

linalg.init()
from Additional._Internal import split_input_tensor, split_output, parse_code


class Layer:
    def __init__(self, input_size: int, output_size: int, activation_function):

        self.weights = gpuarray.to_gpu(numpy.random.normal(0.0, pow(output_size, -0.5), (output_size, input_size)))

        self.input_size = input_size

        self.output_size = output_size

        self.bias = numpy.random.normal(-0.5, 0.5, (output_size, 1))

        self.activation_function = activation_function

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
        a_gpu = gpuarray.to_gpu(input_tensor)
        out_gpu = linalg.dot(a_gpu, self.weights) + self.bias
        return self.activation_function(out_gpu)

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

        error = gpuarray.to_gpu(error)
        output = gpuarray.to_gpu(output)

        temp3 = learning_rate * linalg.dot((error * output * (1.0 - output)),
                                           gpuarray.to_gpu(numpy.array(numpy.transpose(input), ndmin=2)))

        self.bias -= learning_rate * 1 * output

        self.weights += temp3

    def calculate_error(self, output_error):
        return linalg.dot(self.weights.T, gpuarray.to_gpu(output_error))

    def get_info(self):
        return ["Layer", self.input_size, self.output_size, self.weights.tolist(), self.function_source]


class Conv2D_Layer(Layer):
    def __init__(self, pattern_input_size, amount_input_pattern, pattern_output_size, activation_function):

        self.plates = []

        for x in range(amount_input_pattern):
            self.plates.append(
                numpy.random.normal(0.0, pow(pattern_output_size, -0.5), (pattern_output_size, pattern_input_size)))

        self.activation_function = lambda x: activation_function(x)

        self.pattern_size = (pattern_output_size, pattern_input_size)

    def forward(self, input_tensor):
        output = []

        if input_tensor.ndim == 2:
            input_tensor = split_input_tensor(input_tensor, self.pattern_size[1], delete_rest=True)
        for count, element in enumerate(self.plates):
            if count == len(input_tensor):
                break
            output.append(self.activation_function(numpy.dot(element, input_tensor[count])))

        return numpy.array(output)

    def calculate_error(self, output_error):
        error = []
        print(output_error)
        if output_error.ndim == 2:
            output_error = split_input_tensor(output_error, self.pattern_size[1], delete_rest=True)
        print(output_error)
        for count, plate in enumerate(self.plates):
            if count == len(output_error):
                break
            error.append(numpy.dot(plate, output_error[count]))

        return error

    def update_weights(self, error, output, input, learning_rate):
        if error.ndim == 2:
            error = split_input_tensor(error, self.pattern_size[0], delete_rest=True)

        if output.ndim != 1:
            output = split_output(output, self.pattern_size[0], delete_rest=True)

        if input.ndim == 2:
            input = split_input_tensor(input.flatten(), self.pattern_size[1], delete_rest=True)
        print("output:", output.ndim, output)
        for count, plate in enumerate(self.plates):
            print(numpy.transpose(input), "\n", (error[count] * output[count] * (1.0 - output[count])))
            self.plates[count] += learning_rate * numpy.dot((error[count] * output[count] * (1.0 - output[count])),
                                                            numpy.transpose(input[count]))
