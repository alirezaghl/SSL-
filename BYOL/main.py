import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from torch.optim import optimizer
import itertools
from dataclasses import dataclass
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

@dataclass
class Config():
    feature_dim = 2048
    hidden_dim = 4096
    output_dim = 256
    n_epochs = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tau = 0.996  

class Encoder(nn.Module):
    def __init__(self, model = 'resnet50'):
        super().__init__()
        if model == 'resnet50':
            self.encoder = models.resnet50(weights=None)

        self.encoder.fc = nn.Identity()
    
    def forward(self, x):
       out = self.encoder(x)
       return out
        
class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.feature_dim, config.hidden_dim)
        self.bn = nn.BatchNorm1d(config.hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(config.hidden_dim, config.output_dim)
    
    def forward(self, x):
        x = self.linear2(self.act(self.bn(self.linear1(x))))
        return x

class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.output_dim, config.hidden_dim)
        self.bn = nn.BatchNorm1d(config.hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(config.hidden_dim, config.output_dim)
    
    def forward(self, x):
        x = self.linear2(self.act(self.bn(self.linear1(x))))
        return x

class BYOL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.online_encoder = Encoder()
        self.online_projector = Projector(config)
        self.predictor = Predictor(config)
        
        self.target_encoder = Encoder()
        self.target_projector = Projector(config)
        
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        self.target_projector.load_state_dict(self.online_projector.state_dict())
        
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
        
    def forward(self, x, y):
        """
        online path
        """
        z1 = self.predictor(self.online_projector(self.online_encoder(x)))
        z1_prime = self.predictor(self.online_projector(self.online_encoder(y)))

        """
        target path
        """
        with torch.no_grad():
            z2 = self.target_projector(self.target_encoder(x))
            z2_prime = self.target_projector(self.target_encoder(y))

        return z1, z2_prime, z1_prime, z2
    
    def update_target(self, tau):
        for target_param, param in zip(self.target_projector.parameters(), self.online_projector.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
        for target_param, param in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def loss_fn(self, x, y):
        z1, z2_prime, z1_prime, z2 = self.forward(x, y)
    
        def regression_loss(x, y):
            norm_x, norm_y = torch.linalg.norm(x, dim=-1, keepdim=True), torch.linalg.norm(y, dim=-1, keepdim=True)
            return -2 * torch.mean(torch.sum(x*y, dim=-1, keepdim=True) / (norm_x * norm_y))
        
        loss = regression_loss(z1, z2_prime)
        loss += regression_loss(z1_prime, z2)
        return loss.mean()
    
    def optimizer_fn(self):
        all_params = itertools.chain(
            self.online_encoder.parameters(), 
            self.online_projector.parameters(),
            self.predictor.parameters()
        )
        opt = torch.optim.Adam(all_params, lr=1e-3)
        return opt

def get_augmentations():
    Tau = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                              transforms.RandomResizedCrop(size=96),
                              transforms.RandomApply([
                              transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
                              transforms.RandomGrayscale(p=0.2),
                              transforms.GaussianBlur(kernel_size=9, sigma=1.0),
                              transforms.RandomSolarize(threshold=0.5, p=0.0),
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  
                              ])
    
    Tau_prime = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                              transforms.RandomResizedCrop(size=96),
                              transforms.RandomApply([
                              transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
                              transforms.RandomGrayscale(p=0.2),
                              transforms.GaussianBlur(kernel_size=9, sigma=0.1),
                              transforms.RandomSolarize(threshold=0.5, p=0.2),
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  
                              ])

    return Tau, Tau_prime



class BYOLDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform_1, transform_2=None):
        self.base_dataset = base_dataset
        self.transform_1 = transform_1
        self.transform_2 = transform_2 if transform_2 is not None else transform_1
    
    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        img1 = self.transform_1(img)
        img2 = self.transform_2(img)
        return (img1, img2), target
    
    def __len__(self):
        return len(self.base_dataset)

def main():
    config = Config()
    
    tau_transform, tau_prime_transform = get_augmentations()
    base_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    byol_dataset = BYOLDataset(base_dataset, transform_1=tau_transform, transform_2=tau_prime_transform)
    byol_loader = DataLoader(byol_dataset, batch_size=256, shuffle=True, num_workers=0, drop_last=True)

    model = BYOL(config).to(config.device)
    optimizer = model.optimizer_fn()

    for epoch in range(config.n_epochs):
        running_loss = 0.0
        batches_since_last_print = 0
        
        for i, ((img1, img2), _) in enumerate(byol_loader):
            img1, img2 = img1.to(config.device), img2.to(config.device)
            
            loss = model.loss_fn(img1, img2)   

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   

            model.update_target(config.tau)
            
            running_loss += loss.item()
            batches_since_last_print += 1
            
            if i % 10 == 0:
                avg_loss = running_loss / batches_since_last_print if batches_since_last_print > 0 else 0
                print(f"[{epoch + 1}, {i + 1}] loss: {avg_loss:.4f}")
                running_loss = 0.0
                batches_since_last_print = 0
    
    return model

if __name__ == "__main__":
    main()