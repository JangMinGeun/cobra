# License: MIT
# Author: Karl Stelzner

import torch
import torch.nn as nn
import torch.distributions as dists
import torch.nn.functional as F
from torch.nn import Flatten
import numpy as np

import torchvision


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class INConvBlock(nn.Module):
    def __init__(self, nin, nout, stride=1, instance_norm=True, act=nn.ReLU(inplace=True)):
        super(INConvBlock, self).__init__()
        self.conv = nn.Conv2d(nin, nout, 3, stride, 1, bias=not instance_norm)
        if instance_norm:
            self.instance_norm = nn.InstanceNorm2d(nout, affine=True)
        else:
            self.instance_norm = None
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        return self.act(x)


class UNet(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, channel_base=64):
        super().__init__()
        self.num_blocks = num_blocks
        self.down_convs = nn.ModuleList()
        cur_in_channels = in_channels
        for i in range(num_blocks):
            self.down_convs.append(double_conv(cur_in_channels,
                                               channel_base * 2**i))
            cur_in_channels = channel_base * 2**i

        self.tconvs = nn.ModuleList()
        for i in range(num_blocks-1, 0, -1):
            self.tconvs.append(nn.ConvTranspose2d(channel_base * 2**i,
                                                  channel_base * 2**(i-1),
                                                  2, stride=2))

        self.up_convs = nn.ModuleList()
        for i in range(num_blocks-2, -1, -1):
            self.up_convs.append(double_conv(channel_base * 2**(i+1), channel_base * 2**i))

        self.final_conv = nn.Conv2d(channel_base, out_channels, 1)

    def forward(self, x):
        intermediates = []
        cur = x
        for down_conv in self.down_convs[:-1]:
            cur = down_conv(cur)
            intermediates.append(cur)
            cur = nn.MaxPool2d(2)(cur)

        cur = self.down_convs[-1](cur)

        for i in range(self.num_blocks-1):
            cur = self.tconvs[i](cur)
            cur = torch.cat((cur, intermediates[-i -1]), 1)
            cur = self.up_convs[i](cur)

        return self.final_conv(cur)


class UNetModify(nn.Module):

    def __init__(self, input_channels: int, num_blocks, filter_start=32, mlp_size: int = 128):
        super(UNetModify, self).__init__()
        c = filter_start
        self.filter_start = filter_start
        self.mlp_size = mlp_size
        if num_blocks == 4:
            self.down = nn.ModuleList([
                INConvBlock(input_channels + 1, c),
                INConvBlock(c, 2 * c),
                INConvBlock(2 * c, 2 * c),
                INConvBlock(2 * c, 2 * c),  # no downsampling
            ])
            self.up = nn.ModuleList([
                INConvBlock(4 * c, 2 * c),
                INConvBlock(4 * c, 2 * c),
                INConvBlock(4 * c, c),
                INConvBlock(2 * c, c)
            ])
        elif num_blocks == 5:
            self.down = nn.ModuleList([
                INConvBlock(4, c),
                INConvBlock(c, c),
                INConvBlock(c, 2 * c),
                INConvBlock(2 * c, 2 * c),
                INConvBlock(2 * c, 2 * c),  # no downsampling
            ])
            self.up = nn.ModuleList([
                INConvBlock(4 * c, 2 * c),
                INConvBlock(4 * c, 2 * c),
                INConvBlock(4 * c, c),
                INConvBlock(2 * c, c),
                INConvBlock(2 * c, c)
            ])
        elif num_blocks == 6:
            self.down = nn.ModuleList([
                INConvBlock(4, c),
                INConvBlock(c, c),
                INConvBlock(c, c),
                INConvBlock(c, 2 * c),
                INConvBlock(2 * c, 2 * c),
                INConvBlock(2 * c, 2 * c),  # no downsampling
            ])
            self.up = nn.ModuleList([
                INConvBlock(4 * c, 2 * c),
                INConvBlock(4 * c, 2 * c),
                INConvBlock(4 * c, c),
                INConvBlock(2 * c, c),
                INConvBlock(2 * c, c),
                INConvBlock(2 * c, c)
            ])
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(4 * 4 * 2 * c, mlp_size), nn.ReLU(inplace=True),
            nn.Linear(mlp_size, mlp_size), nn.ReLU(inplace=True),
            nn.Linear(mlp_size, 4 * 4 * 2 * c), nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(c, 2, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x_down = [x]
        skip = []
        for i, block in enumerate(self.down):
            act = block(x_down[-1])
            skip.append(act)
            if i < len(self.down) - 1:
                act = F.interpolate(act, scale_factor=0.5, mode='nearest', recompute_scale_factor=True)
            x_down.append(act)
        x_up = self.mlp(x_down[-1]).view(batch_size, -1, 4, 4)
        for i, block in enumerate(self.up):
            features = torch.cat([x_up, skip[-1 - i]], dim=1)
            x_up = block(features)
            if i < len(self.up) - 1:
                x_up = F.interpolate(x_up, scale_factor=2.0, mode='nearest')
        return self.final_conv(x_up)



class AttentionNet(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.unet = UNet(num_blocks=conf.num_blocks,
                         in_channels=4,
                         out_channels=2,
                         channel_base=conf.channel_base)
        self.unet_modify = UNetModify(input_channels=3,
                                      num_blocks=conf.num_blocks,
                                      filter_start=32,
                                      )

        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x, log_scope):
        inp = torch.cat((x, log_scope), 1)
        logits = self.unet(inp)
        # logits = self.unet_modify(inp)
        log_alpha = self.log_softmax(logits)
        # output channel 0 represents alpha_k,
        # channel 1 represents (1 - alpha_k).
        log_mask = log_scope + log_alpha[:, 0:1]
        new_log_scope = log_scope + log_alpha[:, 1:2]
        return log_mask, new_log_scope


class EncoderNet(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        for i in range(4):
            width = (width - 1) // 2
            height = (height - 1) // 2

        self.mlp = nn.Sequential(
            nn.Linear(64 * width * height, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 16)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return x


class DecoderNet(nn.Module):
    def __init__(self, height, width, cobra=False):
        super().__init__()
        self.height = height
        self.width = width
        self.convs = nn.Sequential(
            nn.Conv2d(10, 32, 3),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32, 32, 3),
            # # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, 1),
        )
        ys = torch.linspace(-1, 1, self.height + 6)
        xs = torch.linspace(-1, 1, self.width + 6)
        ys, xs = torch.meshgrid(ys, xs)
        coord_map = torch.stack((ys, xs)).unsqueeze(0)
        self.register_buffer('coord_map_const', coord_map)
        self.cobra = cobra

    def forward(self, z):
        z_tiled = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.height + 6, self.width + 6)
        coord_map = self.coord_map_const.repeat(z.shape[0], 1, 1, 1)
        # print("z_tiled:", z_tiled.shape)
        # print(coord_map.shape)
        inp = torch.cat((z_tiled, coord_map), 1)
        if self.cobra:
            # result = self.convs_cobra(inp)
            result = self.convs(inp)
        else:
            result = self.convs(inp)
        return result


class TransitionMLP(nn.Module):
    '''
    Pytorch module for Deep Q Network
    '''

    def __init__(self, input_size):
        '''
        Define your architecture here
        '''
        super().__init__()
        hidden_size = 512
        output_size = 9
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.hidden_layer1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.Relu = nn.ReLU(inplace=True)
        # self.activation = nn.Tanh()

    def forward(self, state):
        '''
        Get Q values for each action given state
        '''
        x = self.Relu(self.layer1(state))
        x = self.Relu(self.hidden_layer1(x))
        x = self.Relu(self.hidden_layer2(x))
        x = self.Relu(self.hidden_layer3(x))
        x = self.layer2(x)
        # x = self.activation(x)
        return x


class Monet(nn.Module):
    def __init__(self, conf, height, width):
        super().__init__()
        self.conf = conf
        self.attention = AttentionNet(conf)
        self.encoder = EncoderNet(height, width)
        self.decoder = DecoderNet(height, width)
        self.transition = TransitionMLP(10)  # 8 + 2
        self.beta = 0.5
        self.gamma = 0.5

    def forward(self, x):
        log_scope = torch.zeros_like(x[:, 0:1])
        log_masks = []
        for i in range(self.conf.num_slots-1):
            log_mask, log_scope = self.attention(x, log_scope)
            log_masks.append(log_mask)
            # print('mask_shape:', mask.shape)
        log_masks.append(log_scope)
        # loss = torch.zeros_like(x[:, 0, 0, 0])

        full_reconstruction = torch.zeros_like(x)
        # p_xs = torch.zeros_like(loss)
        p_xs = []
        # print('p_xs.shape:', p_xs.shape)
        kl_zs = 0.0
        slots = []
        recon_slots = []
        masks_pred = []

        for i, log_mask in enumerate(log_masks):
            z, kl_z = self.__encoder_step(x, log_mask)
            # print('z_shape:', z.shape)
            sigma = self.conf.bg_sigma if i == 0 else self.conf.fg_sigma
            p_x, x_recon, mask_pred = self.__decoder_step(x, z, log_mask, sigma)
            # print('x_recon.shape:', x_recon.shape)
            # print('mask_pred.shape:', mask_pred.shape)
            masks_pred.append(mask_pred)
            # loss = loss - p_x + self.beta * kl_z
            # loss = loss + self.beta * kl_z
            p_xs.append(p_x.unsqueeze(1))
            kl_zs += kl_z.mean(dim=0)


            # full_reconstruction += mask_pred * x_recon  # wrong....
            slots.append(x_recon)
            # recon_slots.append(x_recon)
            # print('mask_pred.shape:', mask_pred.shape)
        # print('len_slots:', len(slots))

        log_masks = torch.cat(log_masks, 1)
        log_masks_pred = torch.cat(masks_pred, 1).log_softmax(dim=1)
        masks = log_masks.exp()
        p_xs = torch.cat(p_xs, dim=1)
        masks_pred = log_masks_pred.exp()
        # print('masks_pred_shape:', masks_pred.shape)

        # print('masks_shape:', masks.shape)
        # recon_slots = torch.stack(recon_slots, 1)
        # print('slots_shape:', slots.shape)

        # calculate loss
        loss = 0.0
        # print('p_xs_shape:', p_xs.shape)
        p_xs = - p_xs.mean(dim=0).sum()
        loss += p_xs + self.beta * kl_zs

        tr_masks = masks.permute(0, 2, 3, 1)
        # nrows = np.prod(tr_masks.shape[:-1])
        # tr_masks = tr_masks.reshape(nrows, -1)

        tr_masks_pred = masks_pred.permute(0, 2, 3, 1)
        # tr_masks_pred = tr_masks_pred.reshape(nrows, -1)
        tr_masks = tr_masks.clamp_min(1e-5)
        tr_masks_pred = tr_masks_pred.clamp_min(1e-5)

        q_masks = dists.Categorical(probs=tr_masks)
        q_masks_pred = dists.Categorical(probs=tr_masks_pred)
        kl_masks = dists.kl_divergence(q_masks, q_masks_pred)
        kl_masks = torch.sum(kl_masks) / len(masks)  # mean...
        # print('px', p_xs.mean().item(),
        #       'kl_z', kl_zs.mean().item(),
        #       'kl masks', kl_masks.mean().item())
        loss += self.gamma * kl_masks
        # loss = loss - torch.sum(torch.log(p_xs), [1, 2, 3])  # pixel sum(last step)
        for i, slot in enumerate (slots):
            full_reconstruction += masks_pred[:, i].unsqueeze(1) * slot
            recon_slots.append(masks_pred[:, i].unsqueeze(1) * slot)
            # print('mask_pred_shape:', masks[:,i].shape)

        slots = torch.stack(slots, 1)
        recon_slots = torch.stack(recon_slots, 1)

        # calculate cobra version loss
        # total_pred_error_avg = total_pred_error
        # future_pred_loss = torch.sum((full_reconstruction - x).pow(2), [1, 2, 3])
        # error_pred_loss = (total_pred_error_avg - future_pred_loss).pow(2).mean()  # -> question... ???
        # for param in self.transition.parameters():
        #     transition_reg_loss += torch.sum(torch.abs(param))
        # total_loss = future_pred_loss.mean()

        return {'loss': loss,
                'masks': masks,
                'masks_pred': masks_pred,
                'reconstructions': full_reconstruction,
                'slots': slots,
                'recon_slots': recon_slots,
                }


    def __encoder_step(self, x, log_mask):
        encoder_input = torch.cat((x, log_mask), 1)
        q_params = self.encoder(encoder_input)
        # means = q_params[:, :8]
        # sigmas = F.softplus(q_params[:, 8:])
        means = torch.sigmoid(q_params[:, :8]) * 6 - 3
        sigmas = torch.sigmoid(q_params[:, 8:]) * 3
        dist = dists.Normal(means, sigmas)
        dist_0 = dists.Normal(0., sigmas)
        z = means + sigmas * torch.randn_like(sigmas)
        # q_z = dist.log_prob(z)
        kl_z = dists.kl_divergence(dist, dists.Normal(0., 1.))
        kl_z = torch.sum(kl_z, 1)
        return z, kl_z

    def __decoder_step(self, x, z, log_mask, sigma):
        decoder_output = self.decoder(z)
        x_recon = torch.sigmoid(decoder_output[:, :3])
        mask_pred = decoder_output[:, 3:]
        dist = dists.Normal(x_recon, sigma)
        # p_x = torch.exp(dist.log_prob(x))
        p_x = dist.log_prob(x)
        # print('p_x.shape', p_x.shape)
        p_x = p_x * torch.exp(log_mask)
        p_x = torch.sum(p_x, [1, 2, 3])  # -> pixel_sum is last step..
        # print('after_p_x.shape', p_x.shape)
        return p_x, x_recon, mask_pred


class Cobra(nn.Module):
    def __init__(self, conf, height, width):
        super().__init__()
        self.conf = conf
        self.attention = AttentionNet(conf)
        self.encoder = EncoderNet(height, width)
        self.decoder = DecoderNet(height, width, cobra=True)
        self.transition = TransitionMLP(10) # 8 + 2
        self.beta = 0.5
        self.gamma = 0.5

    def forward(self, x, x_next, a): # action is added

        for param in self.attention.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.transition.parameters():
            param.requires_grad = True

        log_scope = torch.zeros_like(x[:, 0:1])
        log_scope_next = torch.zeros_like(x[:, 0:1])
        log_masks = []
        log_masks_next = []
        for i in range(self.conf.num_slots - 1):
            log_mask, log_scope = self.attention(x, log_scope)
            log_masks.append(log_mask)

            log_mask_next, log_scope_next = self.attention(x, log_scope_next)
            log_masks_next.append(log_mask_next)
            # print('mask_shape:', mask.shape)
        log_masks.append(log_scope)
        log_masks_next.append(log_scope_next)
        loss = torch.zeros_like(x[:, 0, 0, 0])

        full_reconstruction = torch.zeros_like(x)
        mask_reconstruction = torch.zeros_like(x)
        full_reconstruction_cur = torch.zeros_like(x)
        mask_reconstruction_cur = torch.zeros_like(x)
        # p_xs = torch.zeros_like(loss)
        p_xs = []
        # print('p_xs.shape:', p_xs.shape)
        kl_zs = 0.0
        # transition_reg_loss = torch.zeros_like(loss)
        transition_reg_loss = 0.0
        total_pred_error = torch.zeros_like(loss)
        a = a.view(-1,2)
        # a = a.view(-1,4)
        slots = []
        masks_pred = []
        recon_slots = []
        mask_recon_slots = []

        slots_cur = []
        masks_pred_cur = []
        recon_slots_cur = []
        mask_slots_cur = []

        for i, log_mask in enumerate(log_masks):
            z, kl_z = self.__encoder_step(x, log_mask)  # kl_z를 loss에 포함시
            sigma = self.conf.bg_sigma if i == 0 else self.conf.fg_sigma
            # print('z.shape:', z.shape)
            # print('a.shape:', a.shape)
            p_x_ori, x_recon_ori, mask_pred_ori = self.__decoder_step(x, z, log_mask, sigma)
            masks_pred_cur.append(mask_pred_ori)
            slots_cur.append(x_recon_ori)

            # print('z_shape:', torch.cat((z, a), 1).shape)
            z_ = self.transition(torch.cat((z, a), 1))
            # transition_reg_loss += torch.sum(torch.abs(z_), dim=1)
            z_ = z + z_[:, :-1]  # appendix notation
            # z_ = z_[:, :-1]
            predicted_error = z_[:, -1]
            # print('z_.shape:', z_.shape)
            total_pred_error += predicted_error
            p_x, x_recon, mask_pred = self.__decoder_step(x_next, z_, log_masks_next[i], sigma)  # mask??? -> mask_pred로 고쳐야 할지..
            masks_pred.append(mask_pred)
            # print(mask_pred.unsqueeze(1))
            p_xs.append(p_x.unsqueeze(1))
            # loss += -p_x + self.beta * kl_z
            # loss += self.beta * kl_z
            kl_zs += kl_z.mean(dim=0)
            # full_reconstruction += mask * x_recon
            slots.append(x_recon)



        # masks = torch.cat(masks, 1)
        # tr_masks = masks.permute(0, 2, 3, 1)
        # q_masks = dists.Categorical(probs=tr_masks)
        # q_masks_recon = dists.Categorical(logits=torch.stack(mask_preds, 3))
        # kl_masks = dists.kl_divergence(q_masks, q_masks_recon)
        # kl_masks = torch.sum(kl_masks, [1, 2])
        # print('px', p_xs.mean().item(),
        #       'kl_z', kl_zs.mean().item(),
        #       'kl masks', kl_masks.mean().item())
        # loss += self.gamma * kl_masks
        # loss += -torch.log(p_xs)
        # cobra loss(JMG)
        p_xs = torch.cat(p_xs, dim=1)
        log_masks = torch.cat(log_masks, 1)
        log_masks_pred = torch.cat(masks_pred, 1).log_softmax(dim=1)
        masks = log_masks.exp()
        masks_pred = log_masks_pred.exp()

        for i, slot in enumerate(slots):
            full_reconstruction += masks_pred[:, i].unsqueeze(1) * slot
            recon_slots.append(masks_pred[:, i].unsqueeze(1) * slot)

            mask_reconstruction += masks[:, i].unsqueeze(1) * slot
            mask_recon_slots.append(masks[:, i].unsqueeze(1) * slot)


        slots = torch.stack(slots, 1)
        recon_slots = torch.stack(recon_slots, 1)
        mask_recon_slots = torch.stack(mask_recon_slots, 1)

        log_masks_pred_ori = torch.cat(masks_pred_cur, 1).log_softmax(dim=1)
        masks_pred_cur = log_masks_pred_ori.exp()

        for i, slot in enumerate(slots_cur):
            full_reconstruction_cur += masks_pred_cur[:, i].unsqueeze(1) * slot
            recon_slots_cur.append(masks_pred_cur[:, i].unsqueeze(1) * slot)

            # mask_reconstruction_cur += masks[:, i].unsqueeze(1) * slot
            # mask_slots_cur.append(masks[:, i].unsqueeze(1) * slot)


        slots_cur = torch.stack(slots_cur, 1)
        recon_slots_cur = torch.stack(recon_slots_cur, 1)



        # calculate loss
        log_likelihood_loss = - p_xs.mean(dim=0).sum() + self.beta * kl_zs
        future_pred_loss = torch.sum((full_reconstruction - x_next).pow(2), [1, 2, 3])
        # print("1:", future_pred_loss.shape)
        # print("2:", total_pred_error.shape)
        error_pred_loss = (total_pred_error - future_pred_loss.detach()).pow(2).mean() # There is no exploration policy -> not used.
        for param in self.transition.parameters():
            transition_reg_loss += torch.sum(torch.square(param))
        # total_loss = future_pred_loss.mean() + error_pred_loss + transition_reg_loss
        total_loss = future_pred_loss.mean() + transition_reg_loss + log_likelihood_loss

        return {'loss': total_loss,
                'masks': masks,
                'masks_pred': masks_pred,
                'reconstructions': full_reconstruction,
                'mask_reconstructions': mask_reconstruction,
                'slots': slots,
                'recon_slots': recon_slots,
                'mask_recon_slots': mask_recon_slots,
                'log_likelihood_loss': log_likelihood_loss,
                'mse_loss': future_pred_loss,
                'error_pred_loss': error_pred_loss,
                'reg_loss': transition_reg_loss,

                'cur_masks_pred': masks_pred_cur,
                'cur_reconstructions': full_reconstruction_cur,
                'cur_slots': slots_cur,
                'cur_recon_slots': recon_slots_cur,

                }

    def __encoder_step(self, x, log_mask):
        encoder_input = torch.cat((x, log_mask), 1)
        q_params = self.encoder(encoder_input)
        # means = q_params[:, :8]
        # sigmas = F.softplus(q_params[:, 8:])
        means = torch.sigmoid(q_params[:, :8]) * 6 - 3
        sigmas = torch.sigmoid(q_params[:, 8:]) * 3
        dist = dists.Normal(means, sigmas)
        dist_0 = dists.Normal(0., sigmas)
        z = means + sigmas * torch.randn_like(sigmas)
        # q_z = dist.log_prob(z)
        kl_z = dists.kl_divergence(dist, dists.Normal(0., 1.))
        kl_z = torch.sum(kl_z, 1)
        return z, kl_z

    def __decoder_step(self, x, z, log_mask, sigma):
        decoder_output = self.decoder(z)
        # print('decoder_output:', decoder_output.shape)
        x_recon = torch.sigmoid(decoder_output[:, :3])
        mask_pred = decoder_output[:, 3].unsqueeze(1)
        dist = dists.Normal(x_recon, sigma)
        p_x = dist.log_prob(x)
        p_x = p_x * torch.exp(log_mask)
        p_x = torch.sum(p_x, [1, 2, 3])
        return p_x, x_recon, mask_pred


def print_image_stats(images, name):
    print(name, '0 min/max', images[:, 0].min().item(), images[:, 0].max().item())
    print(name, '1 min/max', images[:, 1].min().item(), images[:, 1].max().item())
    print(name, '2 min/max', images[:, 2].min().item(), images[:, 2].max().item())


