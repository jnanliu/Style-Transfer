# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import PIL.Image as Image
import torchvision.transforms as transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

img_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

## ------------------- load content, style image ------------------ ##
def load_image(img_path) :
    img = Image.open(img_path).convert('RGB')
    img = img_transforms(img)
    img = img.unsqueeze(0)
    return img

## ------------------- save generated image ------------------ ##
def save_image(img) :
    img = img.squeeze(0)
    t_mean = torch.as_tensor(mean, dtype=img.dtype, device=img.device)
    t_std = torch.as_tensor(std, dtype=img.dtype, device=img.device)
    img = img * t_std[:, None, None] + t_mean[:, None, None]
    img = transforms.ToPILImage()(img.cpu())
    img.save(BASE_DIR + "/g_image.jpg")

## ------------------- generated function ------------------ ##
class GeneratedImage(nn.Module) :

    def __init__(self) :
        super(GeneratedImage, self).__init__()
        self.parmas = nn.Parameter(torch.randn(1, 3, 512, 512), requires_grad=True)
        self.init()

    def init(self) :
        for weight in self.parmas :
            nn.init.normal_(weight, 0., 1.)

    def forward(self) :
        return self.parmas

if __name__ == '__main__' :
    img = GeneratedImage().parmas
    save_image(img)