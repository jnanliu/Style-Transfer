# -*- coding: utf-8 -*-
import os
import time
import torch.cuda as cuda
import torch.optim as optim

from Model.functions import StyleTransferModel
from Image.image_process import load_image, save_image, GeneratedImage

max_epochs = 300

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def train() :
    style_img, content_img = \
        load_image(BASE_DIR + "/Image/style.jpg"), load_image(BASE_DIR + "/Image/content.jpg")
    if cuda.is_available() :
        style_img, content_img = style_img.cuda(), content_img.cuda()
    input_img = GeneratedImage()
    model = StyleTransferModel(style_img, content_img)
    content_loss_list, style_loss_list, total_variation_denoising_loss_list = \
        model.generate_layers()

    optimizer = optim.LBFGS(input_img.parameters())

    if cuda.is_available() :
        model = model.cuda()
        input_img = input_img.cuda()

    epochs = [0]
    while epochs[0] < max_epochs :

        def closure() :
            start_time = time.time()

            model(input_img())
            style_score, content_score, total_variation_denoising_score = 0., 0., 0.

            optimizer.zero_grad()
            for sl in style_loss_list :
                style_score += sl.backward()
            for cl in content_loss_list :
                content_score += cl.backward()
            for tl in total_variation_denoising_loss_list :
                total_variation_denoising_score += tl.backward()

            epochs[0] += 1
            if epochs[0] % 10 == 0 :
                print("epoch {:3d}, content loss {:.2f}, style loss {:.2f}, TV loss {:.2f}, {:.2f} sec".format(
                    epochs[0], content_score, style_score, total_variation_denoising_score, time.time() - start_time))
                save_image(input_img.parmas)

            return content_score + style_score + total_variation_denoising_score

        optimizer.step(closure)

if __name__ == '__main__' :
    train()