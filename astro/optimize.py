import numpy as np

GOLDEN_RATIO: float = (np.sqrt(5) - 1) / 2  # golden ratio


def golden_search(a, b, epsilon, min_function, *args):
    print("Golden Ratio Search:", min_function.__name__)

    # INITIAL SEARCH VALUES
    x0 = a
    x1 = b
    x2 = x0 + (x1 - x0) * (1 - GOLDEN_RATIO)
    x3 = x0 + (x1 - x0) * GOLDEN_RATIO
    x_prev = x3 + 10000000  # add a large number so the while loop doesn't skip

    n = 0
    while np.fabs(x3 - x_prev) > epsilon:
        # CHECK VALUES AND UPDATE CONDITIONS
        if min_function(x3, *args) < min_function(x2, *args):
            x_prev = x3
            x0 = x2
            x2 = x3
            x3 = x0 + (x1 - x0) * GOLDEN_RATIO
        else:
            x_prev = x3
            x1 = x3
            x3 = x2
            x2 = x0 + (x1 - x0) * (1 - GOLDEN_RATIO)
        n += 1
    print("\tIterations:", n)
    return x3
