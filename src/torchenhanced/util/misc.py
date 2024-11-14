import torch
from math import prod


def coord_to_flat(coord_tens, dimensions):
    """
    Given a coordinate tensor of size (...,N), which contains
    the coordinates of points in N dimensions, we return a tensor of size (...,1),
    which contains the same coordinates, for the spacetime flattened from N to 1 dimensions.

    Parameters :
    coord_tens : torch.Tensor (..., N)
        The tensor containing the coordinates of an N-dimensional space
    dimensions : N-tuple
        The size of each of the dimensions. For instance, for an image of size H*W, dimensions = (H,W)

    Returns :
    tensor of size (...), with the flattened indices.

    Example :
    For coord_tens of size (3) where T=[1,2,3], representing a point in 3d space (1,2,3), and given dimensions
    3,3,3, it would output 1*3*3+3*2+3=15, which is the location of the coordinate (1,2,3) in the flattened tensor.
    The extra dimensions are just treated as batch dimensions.
    """
    ndims = coord_tens.shape[-1]
    assert len(dimensions) == ndims

    flat_index_tensor = torch.zeros_like(coord_tens[..., 0])

    for dimnum in range(ndims):
        flat_index_tensor += coord_tens[..., dimnum] * prod(dimensions[dimnum + 1 :])

    return flat_index_tensor
