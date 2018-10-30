
import math

def expit(x):
    return 1 / (1 + pow(math.e, -1 * x))


def logit(x):
    return math.log1p(x / (1 - x))

def identity(x):
    return x

def binary_step(x):
    if x < 0:return 0
    else: return 1

def tanH(x):
    return math.tanh(x)

def softsign(x):
    return 1 / (1 + abs(x))

def relu(x):
    if x < 0:return 0
    return x

def leaky_relu(x):
    if x < 0:return x * 0.01
    return x

def elu(x, a):
    if x <= 0: return a*(pow(math.e,x) - 1)
    else: x

def selu(x):
    if x < 0: return 1.0506*(1.67326 * (pow(math.e, x) - 1))
    else: return x

def apl(x):
    return max(0,x)

def softplus(x):
    return math.log10(1 + pow(math.e, x))

def bent_identity(x):
    return ((math.sqrt(pow(x,2)+1) - 1)/2)+x

def silu(x):
    return x * (1 / (1 + pow(math.e, -1 * x)))

def sinusoid(x):
    return math.sin(x)

def sinc(x):
    if x == 0: return 1
    else: return math.sin(x)/x

def gaussian(x):
    return pow(math.e, -1 * pow(x, 2))
