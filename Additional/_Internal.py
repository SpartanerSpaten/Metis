import numpy
import math
# import marshal
import numpy as np


def _fit_to_pattern(pattern_size, data):
    temp = []

    for y in range(pattern_size):
        temp.append(data[y * pattern_size:(y + 1) * pattern_size])

    return temp


def split_input_tensor(input_tensor, pattern_size, delete_rest=False):
    pattern_size = pattern_size * pattern_size

    length = int(len(input_tensor) / pattern_size)

    return_value = []

    for x in range(length):
        data = input_tensor[x * pattern_size:(x + 1) * pattern_size]

        return_value.append(numpy.array(_fit_to_pattern(int(math.sqrt(pattern_size)), data)))
    if delete_rest is False:
        return_value.append(
            numpy.array(_fit_to_pattern(int(math.sqrt(pattern_size)), input_tensor[length * pattern_size:])))

    return return_value


def parse_code(path, line):
    content = open(path).readlines()
    for count, line_code in enumerate(content[line - 1:]):
        if line_code.replace("\n", "") == "":
            return "".join(content[line:count + line - 1])


def generate_function(function_string):
    # data = marshal.dumps(function_string)
    return


def split_output(output, size, delete_rest=False):
    output = output.flatten()
    lenght = int(output.size / size)
    return_value = []
    for x in range(lenght):
        return_value.append(output[size * x:(x + 1) * size])
    if delete_rest is False:
        temp = output[size * lenght:]
        temp += [0] * (lenght - len(temp))
        return_value.append(temp)
    return numpy.array(return_value)


def im2col_indices(x, field_height=3, field_width=3, padding=1, stride=1):
    x_padded = np.zeros((x.shape[0] + (padding * 2), x.shape[0] + (padding * 2)))
    x_padded[padding:x.shape[0] + padding, padding:x.shape[1] + padding] = x
    cal_pad = int(padding / 2)
    input = []
    for count_y in range(cal_pad, int((x.shape[0] + cal_pad) / stride), stride):
        for count_x in range(cal_pad, int((cal_pad + x.shape[1]) / stride), stride):
            temp = range(count_y, count_y + field_height)
            input.append(x_padded[:, temp][count_x:count_x + field_width])

    array = numpy.array(input)

    return array
