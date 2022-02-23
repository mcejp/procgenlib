from typing import Tuple, Union
import numpy as np


# Implements https://en.wikipedia.org/wiki/Diamond-square_algorithm
#
# Some additional background on distribution of elevation in the real world:
# https://www.wolfram.com/language/12/new-in-geography/distribution-of-elevations.html
def diamond_square(
    rng: np.random.Generator,
    square_size: int,
    num_squares: Tuple[int, int],
    primary_scale: Union[float, np.ndarray],
    roughness: Union[float, np.ndarray],
    base_level: float = 0,
):
    sz = square_size
    h = np.zeros((num_squares[0] * sz + 1, num_squares[1] * sz + 1))

    # Cast primary_scale & roughness to 2D arrays
    if not isinstance(primary_scale, np.ndarray):
        primary_scale = np.full_like(h, primary_scale)
    else:
        assert primary_scale.shape == h.shape

    if not isinstance(roughness, np.ndarray):
        roughness = np.full_like(h, roughness)
    else:
        assert roughness.shape == h.shape

    # sample primary_scale at corner positions and use it to scale an exponential distribution
    corner_scale = primary_scale[
        0 : num_squares[0] * sz + 1 : sz, 0 : num_squares[1] * sz + 1 : sz
    ]
    corner_values = base_level + rng.exponential(scale=corner_scale)

    # for displacement, we go for normal distribution
    randoms = primary_scale * roughness * rng.normal(size=primary_scale.shape)

    # start with the corners
    for i, j in np.ndindex((num_squares[0] + 1, num_squares[1] + 1)):
        h[i * sz, j * sz] = corner_values[i, j]

    # the interpolation distance starts at sqrt(2) * sz (diagonal of one square)
    # and diminishes by a factor of sqrt(2) every half-step
    current_scale = np.sqrt(2)

    while sz >= 2:
        assert sz % 2 == 0

        # "diamond" step
        for i, j in np.ndindex(num_squares):
            # sample 4 corners
            c1 = h[i * sz, j * sz]
            c2 = h[i * sz, (j + 1) * sz]
            c3 = h[(i + 1) * sz, (j + 1) * sz]
            c4 = h[(i + 1) * sz, j * sz]
            c = np.mean([c1, c2, c3, c4])

            displacement = current_scale * randoms[i * sz + sz // 2, j * sz + sz // 2]
            h[i * sz + sz // 2, j * sz + sz // 2] = c + displacement

        num_squares = (num_squares[0] * 2, num_squares[1] * 2)
        sz //= 2
        current_scale /= np.sqrt(2)

        # "square" step
        for j in range(0, num_squares[1] + 1):
            if j % 2 == 0:
                irange = range(1, num_squares[0], 2)
            else:
                irange = range(0, num_squares[0] + 1, 2)

            for i in irange:
                # sample 4 directions
                nan = float("NaN")
                c1 = h[(i - 1) * sz, j * sz] if i > 0 else nan
                c2 = h[i * sz, (j - 1) * sz] if j > 0 else nan
                c3 = h[(i + 1) * sz, j * sz] if i < num_squares[0] - 1 else nan
                c4 = h[i * sz, (j + 1) * sz] if j < num_squares[1] - 1 else nan
                c = np.nanmean([c1, c2, c3, c4])

                displacement = current_scale * randoms[i * sz, j * sz]
                h[i * sz, j * sz] = c + displacement

        current_scale /= np.sqrt(2)

    return h
