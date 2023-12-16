import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid


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