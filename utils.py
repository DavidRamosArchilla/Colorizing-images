import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from PIL import Image


class ColirizationDataset(Dataset):
    def __init__(self, files_list, transformation=None) -> None:
        self.images = files_list
        self.transformation = transformation

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transformation(image)

        transformed_image = image.permute(1, 2, 0).numpy()
        # some images seem to be grayscale, this is a quick fix to skip that image when it happens. Otherwise, the model will throw an error
        if transformed_image.shape[2] == 1:
            return self[index + 1] 

        lab_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        l_channel = torch.from_numpy(l_channel).unsqueeze(0) / 50.0 - 1.0
        a_channel = torch.from_numpy(a_channel) / 110.0
        b_channel = torch.from_numpy(b_channel) / 110.0
        ab = torch.stack([a_channel, b_channel], dim=0)

        return l_channel, ab

    def __len__(self):
        return len(self.images)


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(1.0))
        self.register_buffer("fake_label", torch.tensor(0.0))

    def __call__(self, pred, is_real):
        if is_real:
            labels = self.real_label.expand_as(pred)
        else:
            labels = self.fake_label.expand_as(pred)
        loss = nn.BCELoss()(pred, labels)
        return loss


def visualize_results(model, val_loader, device, epoch, writer=None, show=False, num_images=8):
    model.eval()
    L, true_ab = next(iter(val_loader))
    L, true_ab = L.to(device), true_ab.to(device)

    with torch.no_grad():
        pred_ab = model(L)

    # Convert tensors to images
    L = L.cpu().numpy()
    true_ab = true_ab.cpu().numpy()
    pred_ab = pred_ab.cpu().numpy()
    L = (L + 1) * 50
    pred_ab = pred_ab * 110
    true_ab = true_ab * 110
    true_imgs = []
    pred_imgs = []
    i = 0
    for L, true_ab, pred_ab in zip(L, true_ab, pred_ab):
        i += 1
        true_lab_image = np.concatenate(
            [L.transpose(1, 2, 0), true_ab.transpose(1, 2, 0)], axis=2
        )
        true_rgb_image = cv2.cvtColor(true_lab_image, cv2.COLOR_LAB2RGB)
        true_imgs.append(true_rgb_image.transpose(2, 0, 1))
        pred_lab_image = np.concatenate(
            [L.transpose(1, 2, 0), pred_ab.transpose(1, 2, 0)], axis=2
        )
        pred_rgb_image = cv2.cvtColor(pred_lab_image, cv2.COLOR_LAB2RGB)
        pred_imgs.append(pred_rgb_image.transpose(2, 0, 1))
        if i == num_images:
            break

    true_imgs = np.array(true_imgs)
    pred_imgs = np.array(pred_imgs)

    # Create a grid of images
    imgs = torch.cat([torch.tensor(pred_imgs), torch.tensor(true_imgs)], dim=0)

    # Create a grid of images
    img_grid = make_grid(imgs, nrow=num_images)

    # img = transforms.ToPILImage()(img_grid)
    # img.show()

    # Log to TensorBoard
    if writer:
        writer.add_image("Colorization_Results", img_grid, epoch)

    # Save locally and download
    # add some description, top row are generated images and bottom row are true images 
    plt.figure(figsize=(20, 20))
    plt.figure(figsize=(15, 5))
    plt.imshow(img_grid.permute(1, 2, 0), cmap="viridis")  # .permute(1, 2, 0)
    plt.axis("off")   
    plt.title(f"Colorization Results - Epoch {epoch}")
    if show:
        plt.show()

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Download the image
    # files.download(f'colorization_results_epoch_{epoch}.png')

    plt.close()


def init_weights(net, init='norm', gain=0.02):
    
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
            
    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net

def init_model(model, device):
    model = init_weights(model)
    return model