from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModulationBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ModulationBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 8, 8, 0),
            nn.LeakyReLU(inplace=False),
        )

        self.k_layer = nn.Linear(out_channel, out_channel)
        self.q_layer = nn.Linear(out_channel, out_channel)
        self.v_layer = nn.Linear(out_channel, out_channel)

        self.out_layer = nn.Linear(1024, 1)

    def forward(self, x, simi_mat):
        conv_out = self.conv(x)  # (b, c, 32, 32)
        b, c = conv_out.shape[0], conv_out.shape[1]
        flattened = conv_out.view(b, c, -1)  # (b, c, 1024)
        flattened = flattened.transpose(1, 2)  # (b, 1024, c)
        q = self.q_layer(flattened)
        k = self.k_layer(flattened)
        v = self.v_layer(flattened)
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / sqrt(c)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # -----------------------------------
        num_patches = 1024
        top_k = 32
        simi_mat = simi_mat.to(torch.long)
        # simi_mat[simi_mat == -1] = 0
        row_indices = torch.arange(num_patches).unsqueeze(0).expand(b, num_patches)  # Shape: (B, 1024, 32)
        row_indices = row_indices.unsqueeze(2).expand(b, num_patches, top_k).cuda()

        simi_mat = torch.where(simi_mat == -1, row_indices, simi_mat)

        similarity_weights = torch.arange(1, top_k + 1, dtype=torch.float32)  # Shape: (32,)
        similarity_weights = 1.0 / similarity_weights  # Inverse similarity

        similarity_weights = similarity_weights / similarity_weights.sum()  # Normalize

        similarity_weights = similarity_weights.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 32)
        similarity_weights = similarity_weights.expand(b, num_patches, top_k)  # Shape: (b, 1024, 32)
        similarity_weights = similarity_weights.cuda()

        attention_values = torch.gather(attn_weights, dim=2, index=simi_mat).cuda()  # Shape: (b, 1024, 32)

        weighted_attention_values = attention_values * similarity_weights  # Shape: (b, 1024, 32)

        attn_weights = attn_weights.scatter(2, simi_mat, weighted_attention_values).contiguous()  # Shape: (b, 1024, 1024)
        # -----------------------------------

        attn_out = torch.bmm(attn_weights, v)  # (b, 1024, c)
        transposed = attn_out.transpose(1, 2)  # (b, c, 1024)
        out = self.out_layer(transposed)
        normed_out = F.softmax(out, dim=-1)
        return normed_out


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=False),
        )

        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out


class SSA(nn.Module):
    def __init__(self, in_channel, strides=1):
        super(SSA, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=3, stride=strides, padding=1)
        self.relu1 = nn.LeakyReLU(inplace=False)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=strides, padding=1)
        self.relu2 = nn.LeakyReLU(inplace=False)

        self.conv11 = nn.Conv2d(in_channel, 16, kernel_size=1, stride=strides, padding=0)

    def forward(self, input1, input2):
        '''
        input1 = input1.permute(0, 2, 3, 1)
        input2 = input2.permute(0, 2, 3, 1)
        cat = torch.cat([input1, input2], 3)
        cat = cat.permute(0, 3, 1, 2)
        out1 = self.relu1(self.conv1(cat))
        out1 = self.relu2(self.conv2(out1))
        out2 = self.conv11(cat)
        conv = (out1 + out2).permute(0, 2, 3, 1)
        H, W, K, batch_size = conv.shape[1], conv.shape[2], conv.shape[3],conv.shape[0]
        V = conv.reshape(batch_size,H * W, K)
        Vtrans = torch.transpose(V, 2, 1)
        Vinverse = torch.inverse(torch.bmm(Vtrans, V))
        Projection_map = torch.bmm(V, Vinverse)
        Projection = torch.bmm(Projection_map, Vtrans)
        H1, W1, C1,batch_size = input1.shape[1], input1.shape[2], input1.shape[3], input1.shape[0]
        X1 = input1.reshape(batch_size, H1 * W1, C1)
        Yproj = torch.bmm(Projection, X1)
        Y = Yproj.reshape(batch_size, H1, W1, C1)
        Y = Y.permute(0, 3, 1, 2)
        '''
        K, H, W, batch_size = input2.shape[1], input2.shape[2], input2.shape[3], input2.shape[0]
        cat = torch.cat([input1, input2], 1)
        out1 = self.relu1(self.conv1(cat))
        out1 = self.relu2(self.conv2(out1))
        out2 = self.conv11(cat)
        conv = (out1 + out2)
        Vtrans = conv.reshape(batch_size, 16, H * W)
        Vtrans = Vtrans / (1e-6 + torch.abs(Vtrans).sum(axis=2, keepdims=True))
        V = Vtrans.permute(0, 2, 1)
        Vinverse = torch.inverse(torch.bmm(Vtrans, V))
        Projection = torch.bmm(Vinverse, Vtrans)
        X2 = input2.reshape(batch_size, K, H * W).transpose(2, 1)
        Projection_feature = torch.bmm(Projection, X2)
        Yproj = torch.bmm(V, Projection_feature)
        Y = Yproj.permute(0, 2, 1).reshape(batch_size, K, H, W)

        return Y


class NBNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(NBNet, self).__init__()

        self.in_modulation = ModulationBlock(in_channel, 32)

        self.ConvBlock1 = ConvBlock(1, 32, strides=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.skip1 = nn.Sequential(ConvBlock(32, 32, strides=1), ConvBlock(32, 32, strides=1),
                                   ConvBlock(32, 32, strides=1), ConvBlock(32, 32, strides=1))
        self.ssa1 = SSA(64, strides=1)

        self.ConvBlock2 = ConvBlock(32, 64, strides=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.skip2 = nn.Sequential(ConvBlock(64, 64, strides=1), ConvBlock(64, 64, strides=1),
                                   ConvBlock(64, 64, strides=1))
        self.ssa2 = SSA(128, strides=1)

        self.ConvBlock3 = ConvBlock(64, 128, strides=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.skip3 = nn.Sequential(ConvBlock(128, 128, strides=1), ConvBlock(128, 128, strides=1))
        self.ssa3 = SSA(256, strides=1)

        self.ConvBlock4 = ConvBlock(128, 256, strides=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.skip4 = nn.Sequential(ConvBlock(256, 256, strides=1))
        self.ssa4 = SSA(512, strides=1)

        self.ConvBlock5 = ConvBlock(256, 512, strides=1)

        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.ConvBlock6 = ConvBlock(512, 256, strides=1)

        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.ConvBlock7 = ConvBlock(256, 128, strides=1)

        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.ConvBlock8 = ConvBlock(128, 64, strides=1)

        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.ConvBlock9 = ConvBlock(64, 32, strides=1)

        self.conv10 = nn.Conv2d(32, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x, simi_mat):
        modulation_weights = self.in_modulation(x, simi_mat)
        x = x[:, :1, :, :]

        conv1 = self.ConvBlock1(x)
        conv1 = conv1 * modulation_weights.unsqueeze(3)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.ConvBlock5(pool4)

        up6 = self.upv6(conv5)
        skip4 = self.skip4(conv4)
        skip4 = self.ssa4(skip4, up6)
        up6 = torch.cat([up6, skip4], 1)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        skip3 = self.skip3(conv3)
        skip3 = self.ssa3(skip3, up7)
        up7 = torch.cat([up7, skip3], 1)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        skip2 = self.skip2(conv2)
        skip2 = self.ssa2(skip2, up8)
        up8 = torch.cat([up8, skip2], 1)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        skip1 = self.skip1(conv1)
        skip1 = self.ssa1(skip1, up9)
        up9 = torch.cat([up9, skip1], 1)
        conv9 = self.ConvBlock9(up9)

        conv10 = self.conv10(conv9)
        out = x + conv10

        return out
