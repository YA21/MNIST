import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

NUM_WORKERS = 2


def load_data(data_dir, batch_size):

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    trainset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
    )

    testset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )

    classes = tuple(np.linspace(0, 9, num=10, dtype=np.uint8))

    return trainloader, testloader, classes
