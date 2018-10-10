import random
import numpy
import math
import marshal


def expit(x):
    return 1 / (1 + pow(math.e, -1 * x))


def logit(x):
    return math.log1p(x / (1 - x))


def round(data):
    new_data = []
    for element in data:
        if element > 0.5:
            new_data.append(1)
        else:
            new_data.append(0)
    return new_data


def randomise(data):
    new_data = []

    for element in data:
        new_data.insert(random.randint(0, len(new_data)), element)

    return new_data


def extend_image(data, lengh):
    if len(data) < lengh:
        data += [0] * (lengh - len(data))
    elif len(data) > lengh:
        return data[:lengh]
    return data


def parse_to_image_data(data):
    try:
        length = len(data)
    except:
        length = data.size
        data = data.tolist()

    ratio = int(length / 3)
    ret = []
    for x in range(ratio):
        y = data[x * 3:(x + 1) * 3]
        ret.append((y[0], y[1], y[2]))
    return ret


def _fit_to_pattern(pattern_size, data):
    temp = []

    for y in range(pattern_size):
        temp.append(data[y * pattern_size:(y + 1) * pattern_size])

    return temp


def parse_code(path, line):
    content = open(path).readlines()
    for count, line_code in enumerate(content[line - 1:]):
        if line_code.replace("\n", "") == "":
            return "".join(content[line:count + line - 1])


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


def generate_function(function_string):
    data = marshal.dumps(function_string)

    return data
