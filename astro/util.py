import math

stumpff_range = 20


def stumpff_s(z):
    # Stumpff S function expansion
    s_n = 0
    for i in range(stumpff_range):
        s_n += math.pow(-z, i) / math.factorial(3 + i * 2)
    return s_n


def stumpff_c(z):
    # Stumpff C function expansion
    c_n = 0
    for i in range(stumpff_range):
        c_n += math.pow(-z, i) / math.factorial(2 + i * 2)
    return c_n
