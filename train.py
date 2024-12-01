import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models import ColorizingNet
from utils import ColirizationDataset, init_model
from PIL import Image



if __name__ == "__main__":

    batch_size = 16
    num_epochs = 150
    learning_rate = 2e-4

    # Dataloaders
    transformation_train = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip()
    ])
    transformation_val = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
    ])
    train_dir = 'data/train2017'
    val_dir = 'data/val2017'
    num_train_images = 12000
    num_val_images = 1000
    imgs_train = np.random.choice(os.listdir(train_dir), num_train_images).tolist()
    imgs_train = [os.path.join(train_dir, img) for img in imgs_train]
    imgs_val = np.random.choice(os.listdir(val_dir), num_val_images).tolist()
    imgs_val = [os.path.join(val_dir, img) for img in imgs_val]
    train_dataset = ColirizationDataset(imgs_train, transformation_train)
    val_dataset = ColirizationDataset(imgs_val, transformation_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ColorizingNet(device)
    init_model(model, device)
    print("Model architecture:\n", model)
    print("Number of parameters of the model", sum(p.numel() for p in model.parameters()))
    print("Number of parameters of the generator", sum(p.numel() for p in model.generator.parameters()))
    print("Number of parameters of the discriminator", sum(p.numel() for p in model.discriminator.parameters()))
    # model = torch.compile(model)
    model.train_model(train_loader, val_loader, num_epochs, learning_rate)
