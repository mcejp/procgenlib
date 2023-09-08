import numpy as np
from numpy.random import Generator, PCG64

from procgenlib.synthesis import diamond_square


def test_diamond_square():
    rng = Generator(PCG64(12345))

    result = diamond_square(
        rng, square_size=8, num_squares=(1, 1), primary_scale=1, roughness=1
    )

    expected_result = np.array(
        [
            [0.184, 0.021, 0.042, 0.884, 1.149, 1.515, 2.891, 2.865, 0.645],
            [0.805, 0.393, 0.548, 0.974, 0.896, 1.971, 2.725, 2.354, 1.500],
            [1.161, 0.433, 0.844, 0.945, 0.745, 1.659, 2.285, 2.596, 1.725],
            [1.280, 0.958, 0.860, 0.766, 1.290, 1.717, 1.477, 1.184, 1.394],
            [1.100, 1.512, 2.081, 1.526, 1.391, 1.163, 1.382, 1.539, 1.079],
            [1.700, 2.038, 2.201, 1.958, 1.579, 1.269, 0.791, 1.042, 0.645],
            [1.908, 2.088, 2.449, 2.014, 1.161, 1.005, 0.266, 0.292, 1.017],
            [2.275, 3.220, 2.214, 2.014, 1.193, 1.225, 0.701, 0.210, 0.568],
            [4.690, 3.528, 2.911, 2.677, 2.817, 1.386, 0.644, 0.631, 0.419],
        ]
    )

    # To update expected_result:
    # print(np.array2string(result, separator=", ", formatter={"float_kind": lambda x: "%.3f" % x}))

    assert np.max(np.abs(result - expected_result)) < 1e-3
