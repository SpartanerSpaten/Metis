


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