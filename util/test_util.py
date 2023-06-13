import torch
from .misc import coord_to_flat


def test_coord_to_flat_batch():
    unfolded = torch.randint(0,100,(3,4,5,6)) #4-dim coord tensor
    folded = unfolded.reshape((-1))

    dimensions = unfolded.shape
    testcoord = torch.tensor([[1,2,3,4],[0,2,3,2]]) # Set of coordinate points

    flatcoord = coord_to_flat(testcoord,dimensions)
    assert flatcoord.shape == (2,)
    res = folded[flatcoord]
    for i in range(testcoord.shape[0]):
        assert (res[i]-unfolded[testcoord[i][0]][testcoord[i][1]][testcoord[i][2]][testcoord[i][3]]).all()==0

def test_coord_to_flat_unbatch():
    unfolded = torch.randint(0,100,(3,4,5,6)) #4-dim coord tensor
    folded = unfolded.reshape((-1))

    dimensions = unfolded.shape
    testcoord = torch.tensor([1,2,3,4]) # Set of coordinate points

    flatcoord = coord_to_flat(testcoord,dimensions)
    assert flatcoord.shape==(), f"Invalid shape : {flatcoord.shape}"
    res = folded[flatcoord]
    
    assert (res-unfolded[testcoord[0]][testcoord[1]][testcoord[2]][testcoord[3]]).all()==0

