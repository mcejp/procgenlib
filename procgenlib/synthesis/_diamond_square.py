from typing import Tuple, Union
import numpy as np


def diamond_square(
    rng: np.random.Generator,
    square_size: int,
    num_squares: Tuple[int, int],
    primary_scale: Union[float, np.ndarray],
    roughness: Union[float, np.ndarray],
    base_level: float = 0,
) -> np.ndarray:
    """
    Generate a fractal terrain using the Diamond-square algorithm.

    This function creates a heightmap of a fractal terrain using the
    `Diamond-square algorithm <https://en.wikipedia.org/wiki/Diamond-square_algorithm>`_.
    The generated terrain is represented as a NumPy array.

    Note the algorithm is known to produce some unnatural patterns in the output;
    see the link above for more details.

    Additional background on distribution of elevation in the real world can be found in the following article:
    `Distribution of Elevations: New in Wolfram Language 12 <https://www.wolfram.com/language/12/new-in-geography/distribution-of-elevations.html>`_.


    :param rng: A NumPy random number generator for reproducible randomness.
    :param square_size: The edge length of the basic square.
    :param num_squares: The number of squares to generate along each axis.
    :param primary_scale: The primary scaling factor(s) for height variation.
        This can be a single float or a NumPy array matching the terrain dimensions.
    :param roughness: The roughness factor(s) for height variation.
        This can be a single float or a NumPy array matching the terrain dimensions.
    :param base_level: The base level height for the terrain. (default 0)

    :return: A 2D NumPy array representing the generated terrain heightmap.
        Its dimensions are ``num_squares[0] * square_size + 1`` by ``num_squares[1] * square_size + 1`` samples.
    :rtype: np.ndarray


    :Example:

    Generate a heightmap with specified parameters:

    .. code-block:: python

       import numpy as np
       from numpy.random import Generator, PCG64
       from procgenlib.synthesis import diamond_square

       # Create a random number generator
       rng = Generator(PCG64(12345))

       # Generate the heightmap
       heightmap = diamond_square(rng,
                                  square_size=8,
                                  num_squares=(1, 1),
                                  primary_scale=1,
                                  roughness=1)
    """

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
