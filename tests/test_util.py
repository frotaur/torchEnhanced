import torch, sys, pathlib

sys.path.append(pathlib.Path(__file__).parent.parent.as_posix())

from src.torchenhanced.util import *
from src.torchenhanced.util.misc import coord_to_flat


def test_coord_to_flat_batch():
    unfolded = torch.randint(0, 100, (3, 4, 5, 6))  # 4-dim coord tensor
    folded = unfolded.reshape((-1))

    dimensions = unfolded.shape
    testcoord = torch.tensor([[1, 2, 3, 4], [0, 2, 3, 2]])  # Set of coordinate points

    flatcoord = coord_to_flat(testcoord, dimensions)
    assert flatcoord.shape == (2,)
    res = folded[flatcoord]
    for i in range(testcoord.shape[0]):
        assert (res[i] - unfolded[testcoord[i][0]][testcoord[i][1]][testcoord[i][2]][testcoord[i][3]]).all() == 0


def test_coord_to_flat_unbatch():
    unfolded = torch.randint(0, 100, (3, 4, 5, 6))  # 4-dim coord tensor
    folded = unfolded.reshape((-1))

    dimensions = unfolded.shape
    testcoord = torch.tensor([1, 2, 3, 4])  # Set of coordinate points

    flatcoord = coord_to_flat(testcoord, dimensions)
    assert flatcoord.shape == (), f"Invalid shape : {flatcoord.shape}"
    res = folded[flatcoord]

    assert (res - unfolded[testcoord[0]][testcoord[1]][testcoord[2]][testcoord[3]]).all() == 0


def test_color_comp_incomplete():
    from PIL import Image
    from torchvision import transforms as t

    # ntar=12
    # imgex=torch.clamp(torch.zeros((1,3,1,1)),0,1)
    # imgex[:,0]=140
    # imgex[:,1]=23
    # imgex[:,2]=23
    # print("==================================================")
    # # print("imgex shape : ",imgex.shape)
    # print("Initial : ",(imgex).squeeze())
    # imgex=convert_to_nbit(imgex/255.,ntar)
    # print("After : ",imgex.squeeze())
    # imgex=decode_from_nbit(imgex,ntar)
    # print("Decoded again : ",imgex.squeeze())

    # imgex= Image.open("Riffelsee.jpg")
    # imgex=t.ToTensor()(imgex)[None]
    # showTens(imgex)
    # imgex=rgb_to_yuv(imgex)

    # imgn = convert_to_nbit(imgex,ntar)
    # imgn = decode_from_nbit(imgn,ntar)

    # print("shape : ",imgex.shape)

    # showTens(yuv_to_rgb(imgn))

    # imgyuv = yuv_to_rgb(rgb_to_yuv(imgex))
    # print(f"BOUNDS : (maxy : {torch.max(imgyuv[:,0])},maxCb= {torch.max(imgyuv[:,1])}, maxCr= {torch.max(imgyuv[:,2])})")
