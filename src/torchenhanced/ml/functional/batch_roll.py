import torch

def batch_roll(X, shifts_x,shifts_y):
    """
        Like torch.roll, but supports a batch of shifts. Only applicable for 2D tensors.
        
        X : (H,W) tensor
        shifts_x : (B,) tensor
        shifts_y : (B,) tensor

        Returns : (B,H,W) tensor, shifted accordingly
    """
    assert shifts_x.shape==shifts_y.shape
    B = shifts_x.shape[0]
    H,W = X.shape


    shifts_x = shifts_x[:,None]
    shifts_y = shifts_y[:,None]

    new_x = (torch.arange(H)[None,:] - shifts_x)%H # (B,H)
    new_y = (torch.arange(W)[None,:] - shifts_y)%W # (B,W)
    
    return X[new_x[:,:,None],new_y[:,None,:]]

if __name__=='__main__':
    X = torch.arange(9).reshape(3,3)
    shifts_x = torch.arange(-1,2)
    shifts_y = torch.arange(-1,2,1)

    print(shifts_x, shifts_y)

    rolled= batch_roll(X,shifts_x,shifts_y)
    print(f'rolled shape {rolled.shape}')
    print(f'rolled result : \n{rolled}')
