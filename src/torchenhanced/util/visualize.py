import matplotlib.pyplot as plt
import os, torch, torchvision.transforms as transf
from torchvision.utils import make_grid
import cv2, numpy as np

@torch.no_grad()
def showTens(tensor, columns=None) :
    """"
        Shows tensor as an image using pyplot.
        Any extra dimensions (*,C,H,W) are treated as batch dimensions.

        Args:
        tensor : (H,W) or (C,H,W) or (*,C,H,W) tensor to display
        columns : number of columns to use for the grid of images (default 8 or less)
    """
    tensor = tensor.detach().cpu()

    if(len(tensor.shape)==2):
        fig = plt.figure()
        plt.imshow(tensor[None,:,:])
        plt.axis('off')
        plt.show()
    elif(len(tensor.shape)==3) :
        fig = plt.figure()
        plt.imshow(tensor.permute((1,2,0)))
        plt.axis('off')
        plt.show()
    elif(len(tensor.shape)==4) :
        # Assume B,C,H,W
        B=tensor.shape[0]
        if(columns is not None):
            numCol=columns
        else :
            numCol=min(8,B)

        fig = plt.figure()
        
        to_show=make_grid(tensor,nrow=numCol,pad_value=0.2 ,padding=3)
        if(tensor.shape[1]==1):
            to_show=to_show.mean(dim=0,keepdim=True)

        plt.imshow(to_show.permute(1,2,0))
        if(tensor.shape[1]==1):
            plt.colorbar()
        plt.axis('off')
        plt.show()
    elif(len(tensor.shape)>4):
        tensor = tensor.reshape((-1,tensor.shape[-3],tensor.shape[-2],tensor.shape[-1])) # assume all batch dimensions
        print("Assuming extra dimension are all batch dimensions, newshape : ",tensor.shape)
        showTens(tensor,columns)
    else :
        raise Exception(f"Tensor shape should be (H,W), (C,H,W) or (*,C,H,W), but got : {tensor.shape} !")

@torch.no_grad()
def saveTens(tensor, folderpath,name="imagetensor",columns=None):
    """
        Saves tensor as a png image using pyplot.
        Any extra dimensions (*,C,H,W) are treated as batch dimensions.

        Args:
        tensor : (H,W) or (C,H,W) or (*,C,H,W) tensor to display
        folderpath : relative path of folder where to save the image
        name : name of the image (do not include extension)
        columns : number of columns to use for the grid of images (default 8 or less)
    """
    tensor = tensor.detach().cpu()
    os.makedirs(folderpath,exist_ok=True)
    if(len(tensor.shape)==2) :
        fig = plt.figure()
        plt.imshow(tensor[None,:,:])
        plt.axis('off')
        plt.savefig(os.path.join(folderpath,f"{name}.png"),bbox_inches='tight')
    if(len(tensor.shape)==3) :
        fig = plt.figure()
        plt.imshow(tensor.permute((1,2,0)))
        plt.axis('off')
        plt.savefig(os.path.join(folderpath,f"{name}.png"),bbox_inches='tight')
    elif(len(tensor.shape)==4) :
        # Assume B,C,H,W
        B=tensor.shape[0]
        if(columns is not None):
            numCol=columns
        else :
            numCol=min(8,B)

        fig = plt.figure()
        
        to_show=make_grid(tensor,nrow=numCol,pad_value=0. ,padding=2)
        if(tensor.shape[1]==1):
            to_show=to_show.mean(dim=0,keepdim=True)

        plt.imshow(to_show.permute(1,2,0))
        if(tensor.shape[1]==1):
            plt.colorbar()

        plt.axis('off')
        plt.savefig(os.path.join(folderpath,f"{name}.png"),bbox_inches='tight')
    elif(len(tensor.shape)>4):
        tensor = tensor.reshape((-1,tensor.shape[-3],tensor.shape[-2],tensor.shape[-1])) # assume all batch dimensions
        print("WARNING : assuming extra dimension are all batch dimensions, newshape : ",tensor.shape)
        saveTens(tensor,folderpath,name,columns)
    else :
        raise Exception(f"Tensor shape should be (H,W), (C,H,W) or (*,C,H,W), but got : {tensor.shape} !")

@torch.no_grad()
def saveTensVideo(tensor,folderpath,name="videotensor",columns=None,fps=30,out_size=800):
    """
        Saves tensor as a video. Accepts both (T,H,W), (T,3,H,W) and (*,T,3,H,W).

        Args:
        tensor : (T,H,W) or (T,3,H,W) or (*,T,3,H,W) video tensor
        folderpath : path to save the video
        name : name of the video
        columns : number of columns to use for the grid of videos (default 8 or less)
        fps : fps of the video (default 30)
        out_size : Width of output video (height adapts to not deform videos) (default 800)
    """
    tensor = tensor.detach().cpu()
    os.makedirs(folderpath,exist_ok=True)

    if(len(tensor.shape)<3):
        raise ValueError(f"Tensor shape should be (T,H,W), (T,3,H,W) or (*,T,3,H,W), but got : {tensor.shape} !")
    elif(len(tensor.shape)==3):
        # add channel dimension
        tensor=tensor[:,None,:,:].expand(-1,3,-1,-1) # (T,3,H,W)
        saveTensVideo(tensor,folderpath,name,columns)
    elif(len(tensor.shape)==4):
        if(tensor.shape[1]==1):
            print('Assuming gray-scale video')
            tensor=tensor.expand(-1,3,-1,-1) # (T,3,H,W)
        assert tensor.shape[1]==3, f"Tensor shape should be (T,H,W), (T,3,H,W) or (*,T,3,H,W), but got : {tensor.shape} !"
        # A single video
        _make_save_video(tensor,folderpath,name,fps)
    elif(len(tensor.shape)==5):
        if(tensor.shape[2]==1):
            print('Assuming gray-scale video')
            tensor=tensor.expand(-1,-1,3,-1,-1)
        assert tensor.shape[2]==3, f"Tensor shape should be (T,H,W), (T,3,H,W) or (*,T,3,H,W), but got : {tensor.shape} !"
        # Assume B,T,3,H,W
        B,T,C,H,W = tensor.shape

        if(columns is not None):
            numCol=columns
        else :
            numCol=min(8,B)


        black_cols = (-B)%numCol
        video_tens = torch.cat([tensor.to('cpu'),torch.zeros(black_cols,T,C,H,W)],dim=0) # (B',T,3,H,W)
        video_tens = transf.Pad(3)(video_tens) # (B',T,3,H+3*2,W+3*2)

        B,T,C,H,W = video_tens.shape
        resize_ratio = out_size/(H*numCol)
        indiv_vid_size = int(H*resize_ratio),int(W*resize_ratio)

        video_tens = video_tens.reshape((B*T,C,H,W)) # (B'*T,3,H,W
        video_tens = transf.Resize(indiv_vid_size,antialias=True)(video_tens) # (B'*T,3,H',W')
        video_tens = video_tens.reshape((B,T,C,indiv_vid_size[0],indiv_vid_size[1])) # (B',T,3,H',W')
        B,T,C,H,W = video_tens.shape

        assert B%numCol==0
        numRows = B//numCol

        video_tens = video_tens.reshape((numRows,numCol,T,C,H,W)) # (numRows,numCol,T,3,H',W')
        video_tens = torch.einsum('nmtchw->tcnhmw',video_tens) # (T,C,numRows,H',numCol,W')
        video_tens = video_tens.reshape((T,C,numRows*H,numCol*W)) # (T,C,numRows*H,numCol*W)

        _make_save_video(video_tens,folderpath,name,fps)
    elif (len(tensor.shape)>5):
        video_tens = tensor.reshape((-1,*tensor.shape[-4:]))
        saveTensVideo(video_tens,folderpath,name,columns,fps,out_size)

@torch.no_grad()
def gridify(tensor,out_size=800,columns=None):
    """
        Makes a grid of images/videos from a batch of images. Accepts (B,*,H,W), 

        Args:
            tensor : (B,*,H,W) tensor
            out_size : size of the output grid
            columns : number of columns of the grid
        
        Returns:
            (*,H',W') tensor, representing the grid of images/videos
    """
    B,H,W = tensor.shape[0],tensor.shape[-2],tensor.shape[-1]
    device = tensor.device
    if(columns is not None):
        numCol=columns
    else :
        numCol=min(8,B)


    black_cols = (-B)%numCol
    tensor = torch.cat([tensor,torch.zeros(black_cols,*tensor.shape[1:],device=device)],dim=0) # (B',*,H,W)
    tensor = transf.Pad(3)(tensor) # (B',*,H+3*2,W+3*2)

    B,H,W = tensor.shape[0],tensor.shape[-2],tensor.shape[-1]
    rest_dim = tensor.shape[1:-2]

    rest_dim_prod = 1
    for dim in rest_dim:
        rest_dim_prod*=dim
    
    resize_ratio = out_size/(H*numCol)

    indiv_vid_size = int(H*resize_ratio),int(W*resize_ratio)
    tensor = tensor.reshape((B,rest_dim_prod,H,W))
    tensor = transf.Resize(indiv_vid_size,antialias=True)(tensor) # (B',rest_dim_prod,H',W')
    B,H,W = tensor.shape[0],tensor.shape[-2],tensor.shape[-1]

    assert B%numCol==0

    numRows = B//numCol

    tensor = tensor.reshape((numRows,numCol,rest_dim_prod,H,W)) # (numRows,numCol,rest_dim_prod,H',W')
    tensor = torch.einsum('nmrhw->rnhmw',tensor) # (rest_prod,numRows,H',numCol,W')
    tensor = tensor.reshape((rest_dim_prod,numRows*H,numCol*W)) # (rest_prod,numRows*H,numCol*W)
    tensor = tensor.reshape((*rest_dim,numRows*H,numCol*W)) # (*,numRows*H,numCol*W)

    return tensor

@torch.no_grad()
def _make_save_video(video_tens,folderpath,name,fps=30):
    """
        Makes a video in mp4 and saves it at the given folderpath, with given name.

        Args:
        video_tens : (T,C,H,W) tensor
        path : path to save the video
    """
    T,C,H,W = video_tens.shape
    output_file = os.path.join(folderpath,f"{name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (W, H))
        
    to_save = (255*video_tens.permute(0,2,3,1).cpu().numpy()).astype(np.uint8)

    for t in range(T):
        frame = to_save[t]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)

    video.release()