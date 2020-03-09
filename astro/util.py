import math

import numpy as np

stumpff_range = 20


def magnitude(vector):
    mag_square = 0
    for elem in vector:
        mag_square += math.pow(elem, 2)
    return math.sqrt(mag_square)


def rotate_2D(theta, vector):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    return np.dot(rotation_matrix, vector)


def normalize(vector, mag=None):
    new_vector = [0, 0, 0]
    if mag is None:
        mag = magnitude(vector)
    for i in range(len(vector)):
        new_vector[i] = vector[i] / mag
    return new_vector


def cross_product(v1, v2):
    vec = [v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]]
    return vec


def dot_product(v1, v2):
    value = 0
    for i in range(len(v1)):
        value += v1[i] * v2[i]
    return value


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
