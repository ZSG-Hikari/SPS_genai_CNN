from torchvision import datasets, transforms
from torch.utils.data import DataLoader

_TFM64 = transforms.Compose([
    transforms.Resize((64, 64)),   # upsample CIFAR-10 to 64x64 to fit SpecCNN
    transforms.ToTensor(),
    transforms.Normalize([0.4914,0.4822,0.4465],[0.2470,0.2435,0.2616])
])

def get_data_loader(data_dir="./data", batch_size=128, train=True, shuffle=True):
    ds = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=_TFM64)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2)
