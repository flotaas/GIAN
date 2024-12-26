#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# References : https://github.com/ohhhyeahhh/SiamGAT
# References :https://github.com/GouravWadhwa/Hypergraphs-Image-Inpainting

import torch
import torch.nn as nn
import torch.nn.functional as F

class HypergraphConv(nn.Module):
    def __init__(
            self,
            in_features=256,
            out_features=256,
            edges=64,
            filters=128,
            apply_bias=True,
            theta1=0.0
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.edges = edges
        self.apply_bias = apply_bias
        self.filters = filters
        self.theta1 = theta1

        self.phi_conv = nn.Conv2d(self.in_features, self.filters, kernel_size=1, stride=1, padding=0)
        self.A_conv = nn.Conv2d(self.in_features, self.filters, kernel_size=1, stride=1, padding=0)
        self.M_conv = nn.Conv2d(self.in_features, self.edges, kernel_size=7, stride=1, padding=3)

        self.weight_2 = nn.Parameter(torch.empty(self.in_features, self.out_features))
        nn.init.xavier_normal_(self.weight_2)

        if apply_bias:
            self.bias_2 = nn.Parameter(torch.empty(1, self.out_features))
            nn.init.xavier_normal_(self.bias_2)

    def forward(self, x):
        _, _, feature_height, feature_width = x.shape
        self.vertices = feature_height * feature_width
        # x: torch.Size([20, 1024, 18, 9])
        phi = self.phi_conv(x)
        phi = torch.permute(phi, (0, 2, 3, 1)).contiguous()
        phi = phi.view(-1, self.vertices, self.filters)

        A = F.avg_pool2d(x, kernel_size=(feature_height, feature_width))  # torch.Size([20, 1024, 1, 1])
        A = self.A_conv(A)  # torch.Size([20, 128, 1, 1])
        A = torch.permute(A, (0, 2, 3, 1)).contiguous()  # torch.Size([20, 1, 1, 128])
        A = torch.diag_embed(A.squeeze())  # checked  # torch.Size([20, 128, 128])

        M = self.M_conv(x)  # torch.Size([20, 256, 18, 9])
        M = torch.permute(M, (0, 2, 3, 1)).contiguous()  # torch.Size([20, 18, 9, 256])
        M = M.view(-1, self.vertices, self.edges)  # torch.Size([20, 162, 256])

        # print(phi.shape, A.shape, M.shape)   torch.Size([20, 162, 128]) torch.Size([20, 128, 128]) torch.Size([20, 162, 256])
        # 162是节点数，phi的shape实际上是bs x 节点数N x 特征维度D
        # A的shape实际上是bs x 特征维度D x 特征维度D； M的shape实际上是 bs x 节点数N x M=256？
        H = torch.matmul(phi, torch.matmul(A, torch.matmul(phi.transpose(1, 2), M)))
        H = torch.abs(H)

        if self.theta1 != 0.0:
            mean_H = self.theta1 * torch.mean(H, dim=[1, 2], keepdim=True)
            H = torch.where(H < mean_H, 0.0, H)
        # print(H)    # bs x N x M   torch.Size([20, 162, 256])

        D = H.sum(dim=2)
        D_H = torch.mul(torch.unsqueeze(torch.pow(D + 1e-10, -0.5), dim=-1), H)
        B = H.sum(dim=1)
        B = torch.diag_embed(torch.pow(B + 1e-10, -1))
        x_ = torch.permute(x, (0, 2, 3, 1)).contiguous()
        features = x_.view(-1, self.vertices, self.in_features)  # bs x 节点数N x D

        out = features - torch.matmul(D_H, torch.matmul(B, torch.matmul(D_H.transpose(1, 2), features)))
        out = torch.matmul(out, self.weight_2)

        if self.apply_bias:
            out = out + self.bias_2
        out = torch.permute(out, (0, 2, 1)).contiguous()
        out = out.view(-1, self.out_features, feature_height, feature_width)

        return out


class Graph_Attention_Union(nn.Module):
    def __init__(self, in_channel, out_channel, meanw):
        super(Graph_Attention_Union, self).__init__()
        self.meanw = meanw

        # search region nodes linear transformation
        self.support = nn.Conv2d(out_channel, in_channel, 1, 1)

        # target template nodes linear transformation
        self.query = nn.Conv2d(in_channel, in_channel, 1, 1)

        # linear transformation for message passing
        self.g = nn.Sequential(
            nn.Conv2d(out_channel, in_channel, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        self.init_weights()

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, zf, xf):  # 最后的输出shape和xf一致
        # linear transformation
        xf_trans = self.query(xf)
        zf_trans = self.support(zf)

        # linear transformation for message passing
        # xf_g = self.g(xf)
        zf_g = self.g(zf)

        # calculate similarity
        shape_x = xf_trans.shape
        shape_z = zf_trans.shape

        zf_trans_plain = zf_trans.view(-1, shape_z[1], shape_z[2] * shape_z[3])
        zf_g_plain = zf_g.view(-1, shape_z[1], shape_z[2] * shape_z[3]).permute(0, 2, 1)
        xf_trans_plain = xf_trans.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)
        #print(zf_trans_plain.shape, zf_g_plain.shape, xf_trans_plain.shape)  # torch.Size([4, 1024, 162]) torch.Size([4, 162, 1024]) torch.Size([4, 162, 1024])

        similar = torch.matmul(xf_trans_plain, zf_trans_plain)
        similar = F.softmax(similar, dim=2)
        if self.meanw != 0.0:
            mean_ = torch.mean(similar, dim=[2], keepdim=True)
            similar = torch.where(similar > self.meanw*mean_, similar, 0)
        embedding = torch.matmul(similar, zf_g_plain).permute(0, 2, 1)  # torch.Size([4, 1024, 162])
        embedding = embedding.view(-1, shape_x[1], shape_x[2], shape_x[3])  # torch.Size([4, 1024, 18, 9])

        return embedding

if __name__ == '__main__':
    x1 = torch.randn([4, 32, 15, 1])
    x2 = torch.randn([4, 16, 18, 1])
    x3 = torch.randn([4, 96, 20, 1])
    gat = Graph_Attention_Union(16, 32, meanw=1.3)
    out = gat(x1, x2)
    print(out.shape)



