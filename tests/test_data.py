#from tests import _PATH_DATA
import torch
import os.path
import pytest

#@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")


def test_sets_dimensions():
    _PATH_DATA = "data/processed/"
    processed_tensor = torch.load(_PATH_DATA + "processed_tensor.pt")

    train_dataloader = len(processed_tensor["trainloader"].dataset)
    test_dataloader = len(processed_tensor["testloader"].dataset)
    
    print(train_dataloader)
   
    N=55000

    assert train_dataloader > test_dataloader, "Validation set is bigger than train set"
    assert train_dataloader + test_dataloader == N, "There was some data leak in the processing"
