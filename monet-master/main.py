# License: MIT
# Author: Karl Stelzner

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import visdom
import h5py

import os

import model
import model_modify
import datasets
import config
import wandb
import cv2

# vis = visdom.Visdom()
width = 32


def numpify(tensor):
    return tensor.cpu().detach().numpy()

def visualize_masks_ori(imgs, masks, recons):
    # print('recons min/max', recons[:, 0].min().item(), recons[:, 0].max().item())
    # print('recons1 min/max', recons[:, 1].min().item(), recons[:, 1].max().item())
    # print('recons2 min/max', recons[:, 2].min().item(), recons[:, 2].max().item())
    recons = np.clip(recons, 0., 1.)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    colors.extend([(c[0]//2, c[1]//2, c[2]//2) for c in colors])
    colors.extend([(c[0]//4, c[1]//4, c[2]//4) for c in colors])
    seg_maps = np.zeros_like(imgs)
    masks = np.argmax(masks, 1)
    for i in range(imgs.shape[0]):
        for y in range(imgs.shape[2]):
            for x in range(imgs.shape[3]):
                seg_maps[i, :, y, x] = colors[masks[i, y, x]]

    seg_maps /= 255.0
    vis.images(np.concatenate((imgs, seg_maps, recons), 0), nrow=imgs.shape[0])

def visualize_masks(imgs, masks, recons, slots, loss, conf, step, name='current_img'):
    # print('recons min/max', recons[:, 0].min().item(), recons[:, 0].max().item())
    # print('recons1 min/max', recons[:, 1].min().item(), recons[:, 1].max().item())
    # print('recons2 min/max', recons[:, 2].min().item(), recons[:, 2].max().item())

    recons = np.clip(recons, 0., 1.)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    colors.extend([(c[0]//2, c[1]//2, c[2]//2) for c in colors])
    colors.extend([(c[0]//4, c[1]//4, c[2]//4) for c in colors])
    # print('len(colors):', len(colors))
    seg_maps = np.zeros_like(imgs)
    masks = np.argmax(masks, 1)
    for i in range(imgs.shape[0]):
        for y in range(imgs.shape[2]):
            for x in range(imgs.shape[3]):
                seg_maps[i, :, y, x] = colors[masks[i, y, x]]

    seg_maps /= 255.0
    imgs = np.transpose(imgs, (0,2,3,1))
    seg_maps = np.transpose(seg_maps, (0,2,3,1))
    recons = np.transpose(recons, (0,2,3,1))
    slots = np.transpose(slots, (0,1,3,4,2))
    slots = slots.reshape(8,-1,width,3)
    imgs_concat = cv2.hconcat([list for list in imgs])
    seg_maps_concat = cv2.hconcat([list for list in seg_maps])
    slots_concat = cv2.hconcat([slot for slot in slots])
    recons_concat = cv2.hconcat([list for list in recons])
    watch_img = cv2.vconcat([imgs_concat, seg_maps_concat, slots_concat, recons_concat])
    watch_img = np.array(watch_img)
    watch_img = np.clip(255 * watch_img, 0, 255).astype(np.uint8)
    wandb.log({name: wandb.Image(watch_img)}, step=step)
    wandb.log({name + "_loss": loss}, step=step)
    # vis.images(np.concatenate((imgs, seg_maps, recons), 0), nrow=imgs.shape[0])


def run_training(monet, conf, trainloader):
    if conf.load_parameters and os.path.isfile(conf.checkpoint_file):
        monet.load_state_dict(torch.load(conf.checkpoint_file))
        print('Restored parameters from', conf.checkpoint_file)
    else:
        for w in monet.parameters():
            std_init = 0.01
            nn.init.normal_(w, mean=0., std=std_init)
        print('Initialized parameters')

    optimizer = optim.RMSprop(monet.parameters(), lr=1e-4)

    for epoch in range(conf.num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, counts = data
            images = images.cuda()
            # print('images:', images.dtype)
            # print('images_tensor:', images[0][0][32])
            optimizer.zero_grad()
            output = monet(images)
            loss = torch.mean(output['loss'])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % conf.vis_every == conf.vis_every-1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / conf.vis_every))
                visualize_masks(numpify(images[:8]),
                                # numpify(output['masks'][:8]),
                                numpify(output['masks_pred'][:8]),
                                numpify(output['reconstructions'][:8]),
                                # numpify(output['slots'][:8]),
                                numpify(output['recon_slots'][:8]),
                                running_loss / conf.vis_every,
                                conf,
                                epoch * 90000 + i)
                running_loss = 0.0
        torch.save(monet.state_dict(), conf.checkpoint_file)

    print('training done')


def run_training_cobra(cobra, conf, trainloader):
    if conf.load_parameters and os.path.isfile(conf.checkpoint_file):
        cobra.load_state_dict(torch.load(conf.checkpoint_file))
        print('Restored parameters from', conf.checkpoint_file)
    else:
        for w in cobra.parameters():
            std_init = 0.01
            nn.init.normal_(w, mean=0., std=std_init)
        print('Initialized parameters')

    optimizer = optim.Adam(cobra.parameters(), lr=3e-4)
    device = torch.device('cuda')
    cobra = cobra.to(device)
    for epoch in range(conf.num_epochs):
        running_loss = 0.0

        mse_loss = 0
        log_likelihood_loss = 0
        reg_loss = 0
        error_pred_loss = 0
        for i, data in enumerate(trainloader, 0):
            images, next_images, action, counts = data
            # print(images.shape)
            images = images.cuda()
            next_images = next_images.cuda()
            action = action.cuda()
            optimizer.zero_grad()
            output = cobra(images, next_images, action)
            loss = torch.mean(output['loss'])
            loss.backward()
            optimizer.step()

            mse_loss += torch.mean(output['mse_loss']).item()
            log_likelihood_loss += torch.mean(output['log_likelihood_loss']).item()
            reg_loss += torch.mean(output['reg_loss']).item()
            error_pred_loss += torch.mean(output['error_pred_loss']).item()

            running_loss += loss.item()

            if i % conf.vis_every == conf.vis_every-1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / conf.vis_every))
                print('mse_loss: %.3f, log_likelihood_loss: %.3f, reg_loss: %.3f, error_pred_loss: %.3f' %
                      (mse_loss / conf.vis_every, log_likelihood_loss / conf.vis_every,
                       reg_loss / conf.vis_every, error_pred_loss / conf.vis_every))
                visualize_masks(numpify(next_images[:8]),
                                # numpify(output['masks'][:8]),
                                numpify(output['masks_pred'][:8]),
                                numpify(output['reconstructions'][:8]),
                                # numpify(output['slots'][:8]),
                                numpify(output['recon_slots'][:8]),
                                running_loss / conf.vis_every,
                                conf,
                                epoch * 90000 + i,
                                name='next_img_mask_pred')

                visualize_masks(numpify(images[:8]),
                                numpify(output['cur_masks_pred'][:8]),
                                numpify(output['cur_reconstructions'][:8]),
                                numpify(output['cur_recon_slots'][:8]),
                                running_loss / conf.vis_every,
                                conf,
                                epoch * 90000 + i,
                                name='current_img_mask_pred')

                visualize_masks(numpify(next_images[:8]),
                                numpify(output['masks'][:8]),
                                numpify(output['mask_reconstructions'][:8]),
                                numpify(output['mask_recon_slots'][:8]),
                                running_loss / conf.vis_every,
                                conf,
                                epoch * 90000 + i,
                                name='next_img_mask')
                running_loss = 0.0

                mse_loss = 0
                log_likelihood_loss = 0
                reg_loss = 0
                error_pred_loss = 0
        torch.save(cobra.state_dict(), conf.save_file)

    print('training done')


def sprite_experiment():
    conf = config.sprite_config
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.float()),
                                    ])
    trainset = datasets.Sprites(conf.data_dir, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=conf.batch_size,
                                              shuffle=True, num_workers=2)
    monet = model_modify.Monet(conf, 32, 32).cuda()
    if conf.parallel:
        monet = nn.DataParallel(monet)
    run_training(monet, conf, trainloader)


def cobra_experiment():
    conf = config.cobra_config

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.float()),
                                    ])
    trainset = datasets.Cobra(conf.data_dir, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=conf.batch_size,
                                              shuffle=True, num_workers=2)
    cobra = model_modify.Cobra(conf, 32, 32).cuda()
    if conf.parallel:
        cobra = nn.DataParallel(cobra)
    run_training_cobra(cobra, conf, trainloader)


def clevr_experiment():
    conf = config.clevr_config
    # Crop as described in appendix C
    crop_tf = transforms.Lambda(lambda x: transforms.functional.crop(x, 29, 64, 192, 192))
    drop_alpha_tf = transforms.Lambda(lambda x: x[:3])
    transform = transforms.Compose([crop_tf,
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    drop_alpha_tf,
                                    transforms.Lambda(lambda x: x.float()),
                                   ])
    trainset = datasets.Clevr(conf.data_dir,
                              transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=conf.batch_size,
                                              shuffle=True, num_workers=8)
    monet = model.Monet(conf, 128, 128).cuda()
    if conf.parallel:
        monet = nn.DataParallel(monet)
    run_training(monet, conf, trainloader)


if __name__ == '__main__':
    # clevr_experiment()
    # sprite_experiment()
    conf = config.sprite_config
    dict_conf = conf._asdict()
    # print('conf:', conf._asdict())
    wandb.init(project="Cobra",
               name='run_monet_elastic',
               # name='run_cobra',
               config=dict(dict_conf))
    # cobra_experiment()
    sprite_experiment()
