import torchvision.datasets as td
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from pathlib import Path

data_folder = str(Path(__file__).resolve().parent.parent / "data")

def get_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1,1]
    ])
    train = td.CIFAR10(root=data_folder, train=True, transform=transform, download=True)
    test = td.CIFAR10(root=data_folder, train=False, transform=transform, download=True)
    train, val = train_test_split(train, test_size=0.1)
    return train, val, test


def get_stl10():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize STL-10 to 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train = td.STL10(root=data_folder, split='train', transform=transform, download=True)
    test = td.STL10(root=data_folder, split='test', transform=transform, download=True)
    train, val = train_test_split(train, test_size=0.1)
    return train, val, test

def get_celeba():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    dataset = td.CelebA(root=data_folder, split="train", transform=transform, download=True)
    train, val = train_test_split(dataset, test_size=0.1)
    return train, val
