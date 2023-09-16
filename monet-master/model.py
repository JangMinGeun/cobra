# License: MIT
# Author: Karl Stelzner

import torch
import torch.nn as nn
import torch.distributions as dists
import torch.nn.functional as F

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


class AttentionNet(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.unet = UNet(num_blocks=conf.num_blocks,
                         in_channels=4,
                         out_channels=2,
                         channel_base=conf.channel_base)

    def forward(self, x, scope):
        inp = torch.cat((x, scope), 1)
        logits = self.unet(inp)
        alpha = torch.softmax(logits, 1)
        # output channel 0 represents alpha_k,
        # channel 1 represents (1 - alpha_k).
        mask = scope * alpha[:, 0:1]
        new_scope = scope * alpha[:, 1:2]
        return mask, new_scope

class Flatten(nn.Module):

    def forward(self, x):
        return x.flatten(start_dim=1)


class EncoderNet(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(inplace=True)
        )

        for i in range(4):
            width = (width - 1) // 2
            height = (height - 1) // 2

        self.mlp = nn.Sequential(
            nn.Linear(64 * width * height, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 32)
        )

    def forward(self, x):
        x = self.convs(x)
        # print('x.shape:', x.shape)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x) # shape = 64, 32
        return x


class DecoderNet(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.convs = nn.Sequential(
            nn.Conv2d(18, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, 1),
        )
        ys = torch.linspace(-1, 1, self.height + 8)
        xs = torch.linspace(-1, 1, self.width + 8)
        ys, xs = torch.meshgrid(ys, xs)
        coord_map = torch.stack((ys, xs)).unsqueeze(0)
        self.register_buffer('coord_map_const', coord_map)

    def forward(self, z):
        z_tiled = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.height + 8, self.width + 8)
        coord_map = self.coord_map_const.repeat(z.shape[0], 1, 1, 1)
        inp = torch.cat((z_tiled, coord_map), 1)
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
        output_size = 17
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.Relu = nn.ReLU(inplace=True)
        # self.activation = nn.Tanh()

    def forward(self, state):
        '''
        Get Q values for each action given state
        '''
        x = self.Relu(self.layer1(state))
        x = self.Relu(self.layer2(x))
        x = self.layer3(x)
        # x = self.activation(x)
        return x

class Monet(nn.Module):
    def __init__(self, conf, height, width):
        super().__init__()
        self.conf = conf
        self.attention = AttentionNet(conf)
        self.encoder = EncoderNet(height, width)
        self.decoder = DecoderNet(height, width)
        self.beta = 0.5
        self.gamma = 0.5

    def forward(self, x):
        scope = torch.ones_like(x[:, 0:1])
        masks = []
        for i in range(self.conf.num_slots-1):
            mask, scope = self.attention(x, scope)
            masks.append(mask)
            # print('mask_shape:', mask.shape)
        masks.append(scope)
        loss = torch.zeros_like(x[:, 0, 0, 0])
        mask_preds = []
        full_reconstruction = torch.zeros_like(x)
        p_xs = torch.zeros_like(loss)
        # p_xs = torch.zeros_like(x)
        # print('p_xs.shape:', p_xs.shape)
        kl_zs = torch.zeros_like(loss)
        slots = []
        recon_slots = []
        for i, mask in enumerate(masks):
            z, kl_z = self.__encoder_step(x, mask)
            # print('z_shape:', z.shape)
            sigma = self.conf.bg_sigma if i == 0 else self.conf.fg_sigma
            p_x, x_recon, mask_pred = self.__decoder_step(x, z, mask, sigma)  # mask별로 x_recon을 만드는 것이 옳은가? O
            # print('x_recon.shape:', x_recon.shape)
            # print('mask_pred.shape:', mask_pred.shape)
            mask_preds.append(mask_pred)
            loss = loss - p_x + self.beta * kl_z
            # loss = loss + self.beta * kl_z
            p_xs += -p_x
            kl_zs += kl_z
            full_reconstruction += mask * x_recon
            slots.append(mask * x_recon)
            recon_slots.append(x_recon)
            # print('mask_pred.shape:', mask_pred.shape)
        # print('len_slots:', len(slots))

        masks = torch.cat(masks, 1)
        # print('masks_shape:', masks.shape)
        slots = torch.stack(slots, 1)
        recon_slots = torch.stack(recon_slots, 1)
        # print('slots_shape:', slots.shape)
        tr_masks = masks.permute(0, 2, 3, 1)
        q_masks = dists.Categorical(probs=tr_masks)
        q_masks_recon = dists.Categorical(logits=torch.stack(mask_preds, 3))
        kl_masks = dists.kl_divergence(q_masks, q_masks_recon)
        kl_masks = torch.sum(kl_masks, [1, 2])
        # print('px', p_xs.mean().item(),
        #       'kl_z', kl_zs.mean().item(),
        #       'kl masks', kl_masks.mean().item())
        loss = loss + self.gamma * kl_masks
        # loss = loss - torch.sum(torch.log(p_xs), [1, 2, 3])  # pixel sum(last step)
        return {'loss': loss,
                'masks': masks,
                'reconstructions': full_reconstruction,
                'slots': slots,
                'recon_slots': recon_slots}


    def __encoder_step(self, x, mask):
        encoder_input = torch.cat((x, mask), 1)
        q_params = self.encoder(encoder_input)
        means = torch.sigmoid(q_params[:, :16]) * 6 - 3
        sigmas = torch.sigmoid(q_params[:, 16:]) * 3
        dist = dists.Normal(means, sigmas)
        dist_0 = dists.Normal(0., sigmas)
        z = means + dist_0.sample()
        q_z = dist.log_prob(z)
        kl_z = dists.kl_divergence(dist, dists.Normal(0., 1.))
        kl_z = torch.sum(kl_z, 1)
        return z, kl_z

    def __decoder_step(self, x, z, mask, sigma):
        decoder_output = self.decoder(z)
        x_recon = torch.sigmoid(decoder_output[:, :3])
        # x_recon = decoder_output[:, :3]
        mask_pred = decoder_output[:, 3]
        dist = dists.Normal(x_recon, sigma)
        # p_x = torch.exp(dist.log_prob(x))
        p_x = dist.log_prob(x)
        # print('p_x.shape', p_x.shape)
        p_x = p_x * mask
        p_x = torch.sum(p_x, [1, 2, 3])  # -> pixel_sum is last step..
        # print('after_p_x.shape', p_x.shape)
        return p_x, x_recon, mask_pred


class Cobra(nn.Module):
    def __init__(self, conf, height, width):
        super().__init__()
        self.conf = conf
        self.attention = AttentionNet(conf)
        self.encoder = EncoderNet(height, width)
        self.decoder = DecoderNet(height, width)
        self.transition = TransitionMLP(18) # 16 + 2
        self.beta = 0.5
        self.gamma = 0.25

    def forward(self, x, x_next, a): # action is added
        scope = torch.ones_like(x[:, 0:1])
        masks = []
        for i in range(self.conf.num_slots-1):
            mask, scope = self.attention(x, scope)
            masks.append(mask)
        masks.append(scope)
        loss = torch.zeros_like(x[:, 0, 0, 0])
        mask_preds = []
        slots = []
        full_reconstruction = torch.zeros_like(x)
        p_xs = torch.zeros_like(loss)
        kl_zs = torch.zeros_like(loss)
        transition_reg_loss = torch.zeros_like(loss)
        total_pred_error = torch.zeros_like(loss)
        a = a.view(-1, 2)
        # print('x.shape:', x.shape)
        # masks = torch.cat(masks, 1)
        for i, mask in enumerate(masks):
            z, kl_z = self.__encoder_step(x, mask)
            sigma = self.conf.bg_sigma if i == 0 else self.conf.fg_sigma
            # print('z.shape:', z.shape)
            # print('a.shape:', a.shape)
            # print('z_shape:', torch.cat((z, a), 1).shape)
            z_ = self.transition(torch.cat((z, a), 1))
            transition_reg_loss = transition_reg_loss + torch.sum(torch.abs(z_), dim=1)
            z_ = z + z_[:, :-1]
            predicted_error = z_[:, -1]
            total_pred_error += predicted_error
            p_x, x_recon, mask_pred = self.__decoder_step(x, z_, mask, sigma)  # mask??? -> mask_pred로 고쳐야 할지..
            mask_preds.append(mask_pred.unsqueeze(1))
            # print(mask_pred.unsqueeze(1))
            loss += -p_x + self.beta * kl_z
            # loss += self.beta * kl_z
            p_xs += -p_x
            kl_zs += kl_z
            full_reconstruction += mask * x_recon
            slots.append(mask * x_recon)

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
        masks = torch.cat(masks, 1)
        mask_preds = torch.cat(mask_preds, 1)
        slots = torch.stack(slots, 1)
        total_pred_error_avg = total_pred_error
        future_pred_loss = torch.sum((full_reconstruction - x_next).pow(2), [1, 2, 3])
        error_pred_loss = (total_pred_error_avg - future_pred_loss).pow(2) # -> question... ???
        # for param in self.transition.parameters():
        #     transition_reg_loss += torch.sum(torch.abs(param))
        total_loss = future_pred_loss + error_pred_loss + transition_reg_loss


        return {'loss': total_loss,
                'masks': masks,
                'mask_preds': mask_preds,
                'reconstructions': full_reconstruction,
                'slots': slots,}


    def __encoder_step(self, x, mask):
        encoder_input = torch.cat((x, mask), 1)
        q_params = self.encoder(encoder_input)
        means = torch.sigmoid(q_params[:, :16]) * 6 - 3
        sigmas = torch.sigmoid(q_params[:, 16:]) * 3 + 1e-7  # prevent error of bound > 0
        dist = dists.Normal(means, sigmas)
        dist_0 = dists.Normal(0., sigmas)
        z = means + dist_0.sample()
        q_z = dist.log_prob(z)
        kl_z = dists.kl_divergence(dist, dists.Normal(0., 1.))
        kl_z = torch.sum(kl_z, 1)
        return z, kl_z

    def __decoder_step(self, x, z, mask, sigma):
        decoder_output = self.decoder(z)
        x_recon = torch.sigmoid(decoder_output[:, :3])
        mask_pred = decoder_output[:, 3]
        dist = dists.Normal(x_recon, sigma)
        p_x = dist.log_prob(x)
        p_x = p_x * mask
        p_x = torch.sum(p_x, [1, 2, 3])
        return p_x, x_recon, mask_pred


def print_image_stats(images, name):
    print(name, '0 min/max', images[:, 0].min().item(), images[:, 0].max().item())
    print(name, '1 min/max', images[:, 1].min().item(), images[:, 1].max().item())
    print(name, '2 min/max', images[:, 2].min().item(), images[:, 2].max().item())


