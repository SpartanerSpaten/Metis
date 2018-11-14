
import numpy



class Error:
    def __init__(self):
        self.error = []
        self.layer_typ = []


    @staticmethod
    def calculate(input: numpy.ndarray, desired: numpy.ndarray, output : numpy.ndarray, layer : list) -> 'Error':

        error = Error()

        output_error = desired - output

        for count in range(0, len(layer)):
            index = len(layer) - (count + 1)
            layer = layer[index]
            output_error = layer.calculate_error(output_error)
            error.error.insert(0, output_error)
            error.layer_typ.insert(0,layer.__class__)

        return error

    def __next__(self):

        # Prepare vectors for next layer

        pass