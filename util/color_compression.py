## Includes some function to manipulate image tensors
import torch


def rgb_to_yuv(img) :
    """
        Convert a (B,3,...) RGB [0.,1.] tensor to a (B,3,...) [0,1.] YCbCr tensor.
    """
    img_yuv =torch.zeros_like(img)
    (R,G,B)  = img[:,0]*255,img[:,1]*255,img[:,2]*255
    img_yuv[:,0]=    + 0.299*R +.587*G+.114*B
    img_yuv[:,1]= 128- 0.169*R -0.331 *G+0.5*B
    img_yuv[:,2]= 128+ 0.5  *R -0.419 *G-0.081*B

    return img_yuv/255.

def yuv_to_rgb(img):
    """
        Converts a (B,3,...) YCbCr [0,255] tensor to a (B,3,...) RGB [0.,1.] tensor.
    """
    img_rgb = torch.zeros_like(img)
    (Y,Cb,Cr)  = img[:,0]*255,img[:,1]*255.-128,img[:,2]*255.-128

    img_rgb[:,0]= Y+0*Cb +1.4*Cr
    img_rgb[:,1]= Y-0.343 *Cb-0.711*Cr
    img_rgb[:,2]= Y+1.765 *Cb-0*Cr
    
    return img_rgb/255.

def convert_to_nbit(imgs,n):
    """
        Convert a (B,3,...) 8-bit-channel tensor with values in 0.-1.,
        to a (B,1,...) tensor with nbit-encoded RGB values. For now, n should
        be divisible by 3.
    """
    dtype=torch.int32

    if(n%3!=0):
        raise ValueError(f"n={n} not divisible by 3")

    chn = n//3
    imgs=(imgs*255.).to(dtype=dtype)
    
    # Introduce 0's in front (keep only the topmost chanbits bits)
    R=(imgs[:,0]>>(8-chn))
    G=(imgs[:,1]>>(8-chn))
    B=(imgs[:,2]>>(8-chn))

    img_coded=(R<<chn*2) | (G<<chn) | B

    #Restore 1-dim channel :

    return img_coded[:,None]


def decode_from_nbit(imgnbit,n):
    """
        Given a (B,1,...) n-bit coded tensor (hence with int values,
        in the range 0 to 2^n-1), decodes to a standard (B,3,...) tensor
        with 8-bit coded channels, and values in floats [0.,1.].
    """
    if(n%3!=0):
        raise ValueError(f"n={n} not divisible by 3")
    
    chn=n//3
    dtype=torch.int32

    trailing = torch.tensor(2**(8-chn-1),dtype=dtype) # Trailing bits which where lost. We choose around the middle

    # Erasing the trailing bits, and replaces them with "trailing"
    # We "and" with 255 (0xFF), to keep numbers only up to that value
    R = (((imgnbit >> chn*2) << (8-chn)) & 0xFF) | trailing
    G = ((imgnbit >> chn) << (8-chn)) & 0xFF | trailing
    B = (imgnbit << (8-chn)) & 0xFF | trailing

    return torch.cat([R,G,B],dim=1)/255.






if __name__=="__main__" :
    from PIL import Image
    from torchvision import transforms as t
    from visualize import showTens

    ntar=12
    imgex=torch.clamp(torch.zeros((1,3,1,1)),0,1)
    imgex[:,0]=140
    imgex[:,1]=23
    imgex[:,2]=23
    print("==================================================")
    # print("imgex shape : ",imgex.shape)
    print("Initial : ",(imgex).squeeze())
    imgex=convert_to_nbit(imgex/255.,ntar)
    print("After : ",imgex.squeeze())
    imgex=decode_from_nbit(imgex,ntar)
    print("Decoded again : ",imgex.squeeze())


    imgex= Image.open("Riffelsee.jpg")
    imgex=t.ToTensor()(imgex)[None]
    showTens(imgex)
    imgex=rgb_to_yuv(imgex)
    
    
    imgn = convert_to_nbit(imgex,ntar)
    imgn = decode_from_nbit(imgn,ntar)


    print("shape : ",imgex.shape)

    showTens(yuv_to_rgb(imgn))

    imgyuv = yuv_to_rgb(rgb_to_yuv(imgex))
    print(f"BOUNDS : (maxy : {torch.max(imgyuv[:,0])},maxCb= {torch.max(imgyuv[:,1])}, maxCr= {torch.max(imgyuv[:,2])})")
    # showTens(imgyuv)