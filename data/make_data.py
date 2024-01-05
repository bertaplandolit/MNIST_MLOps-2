import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
import os

def mnist(folder='data/raw/corruptmnist/'):
    files = os.listdir(folder)

    train_images = []
    test_images = []

    train_targets = []
    test_targets = []

    transform_normalize = transforms.Normalize(mean=1, std=1)

    for i in files:
        if 'train_images' in i:
            train_images.append(torch.load(folder+i))

        elif 'train_target' in i:
            train_targets.append(torch.load(folder+i))

        elif 'test_images' in i:
            test_images.append(torch.load(folder+i))
        
        elif 'test_target' in i:
            test_targets.append(torch.load(folder+i))
    
    train_images =torch.concatenate(train_images)
    train_targets =torch.concatenate(train_targets)
    test_images =torch.concatenate(test_images)
    test_targets =torch.concatenate(test_targets)
 
    print('Found {} train images and {} test images'.format(len(train_images), len(test_images)))

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    # Create datasets
    trainset = TensorDataset(train_images, train_targets)
    testset = TensorDataset(test_images, test_targets)

    # Create dataloaders
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=True)    

    results= {'trainloader': trainloader, 'testloader': testloader}

    torch.save(results, 'data/processed/processed_tensor.pt')

    return trainloader, testloader

train, test = mnist()