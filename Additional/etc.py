import random


def round(data):
    new_data = []
    for element in data:
        if element > 0.5:
            new_data.append(1)
        else:
            new_data.append(0)
    return new_data


def randomise(data):
    return random.shuffle(data)




