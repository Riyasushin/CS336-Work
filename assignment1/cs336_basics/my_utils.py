import torch
from typing import IO, Any, BinaryIO
import os

def softmax(in_features: torch.Tensor, dim: int):
    '''
    Given a tensor of inputs, return the output of softmaxing the given `dim` of the input
    '''
    # 调用张量的.max(dim=..., keepdim=...)方法时，返回的是一个元组  (最大值, 索引)
    max_val = in_features.max(dim=dim, keepdim=True).values
    in_features =  in_features - max_val
    exp_x = torch.exp(in_features)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp



'''
nn.optim.Optimizer / nn.Module has a state_dict(), load_state_dict()
torch.save (obj, dest) can dump an object
torch.load(src)
'''

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    iter: int = checkpoint['iter']
    return iter

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": iteration,
    }
    torch.save(checkpoint, out)




    