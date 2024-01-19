import matplotlib.pyplot as plt
import os, torch, torchvision.transforms as transf
from torchvision.utils import make_grid
import cv2, numpy as np

@torch.no_grad()
def showTens(tensor, columns=None) :
    '''
        shows tensor as an image. Accepts (H,W), (C,H,W) and (*,C,H,W).
    '''
    if(len(tensor.shape)==2):
        fig = plt.figure()
        plt.imshow(tensor[None,:,:])
        plt.show()
    elif(len(tensor.shape)==3) :
        fig = plt.figure()
        plt.imshow(tensor.permute((1,2,0)))
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
        print("WARNING : assuming extra dimension are all batch dimensions, newshape : ",tensor.shape)
        showTens(tensor,columns)
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
        assert tensor.shape[2]==3, f"Tensor shape should be (T,H,W), (T,3,H,W) or (*,T,3,H,W), but got : {tensor.shape} !"
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
def saveTensImage(tensor, folderpath,name="imagetensor",columns=None):
    '''
        Saves tensor as an image. Accepts both (C,H,W) and (*,C,H,W). 
    '''
    if(len(tensor.shape)==2) :
        fig = plt.figure()
        plt.imshow(tensor[None,:,:])
        plt.savefig(os.path.join(folderpath,f"{name}.png"))
    if(len(tensor.shape)==3) :
        fig = plt.figure()
        plt.imshow(tensor.permute((1,2,0)))
        plt.savefig(os.path.join(folderpath,f"{name}.png"))
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
        #createGrid(tensor,fig,numCol)
        plt.axis('off')
        plt.savefig(os.path.join(folderpath,f"{name}.png"))
    elif(len(tensor.shape)>4):
        tensor = tensor.reshape((-1,tensor.shape[-3],tensor.shape[-2],tensor.shape[-1])) # assume all batch dimensions
        print("WARNING : assuming extra dimension are all batch dimensions, newshape : ",tensor.shape)
        saveTensImage(tensor,folderpath,name,columns)
    else :
        raise Exception(f"Tensor shape should be (H,W), (C,H,W) or (*,C,H,W), but got : {tensor.shape} !")

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