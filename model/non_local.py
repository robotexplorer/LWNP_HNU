import torch
from torch import nn
from torch.nn import functional as F
class EfficientNL(nn.Module):
    def __init__(self, in_channels, key_channels=None, head_count=None, value_channels=None):
        super(EfficientNL, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        if  self.key_channels==None:
            self.key_channels=self.in_channels//2
        if self.value_channels == None:
            self.value_channels = self.in_channels // 2
        if self.head_count == None:
            self.head_count = 2
        self.keys = nn.Conv3d( self.in_channels, self.key_channels, 1)
        self.queries = nn.Conv3d( self.in_channels,  self.key_channels, 1)
        self.values = nn.Conv3d( self.in_channels,  self.value_channels, 1)
        self.reprojection = nn.Conv3d(self.value_channels,  self.in_channels, 1)

    def forward(self, input_):
        n, _,c, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels,-1))
        queries = self.queries(input_).reshape(n, self.key_channels, -1)
        values = self.values(input_).reshape((n, self.value_channels, -1))
        head_key_channels = self.key_channels // self.head_count#8/2=4
        head_value_channels = self.value_channels // self.head_count#8/2=4

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:,i * head_key_channels: (i + 1) * head_key_channels,:], dim=2)
            query = F.softmax(queries[:,i * head_key_channels: (i + 1) * head_key_channels,:], dim=1)
            value = values[:,i * head_value_channels: (i + 1) * head_value_channels,:]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels,c, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_
        return attention

class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded',
                 dimension=3, bn_layer=True, levels=None):

        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if levels is not None:
            self.ssp=True
            self.p = SpatialPyramidPooling(levels=[2*i+1 for i in range(0,levels)])
        else:
            self.ssp = False
        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 4
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )
        # print()
    def forward(self, x):

        batch_size,c,t,h,w = x.size()

        g_x = self.g(x).view(batch_size, -1, h,w)
        if self.ssp:
            g_x = self.p(g_x)
        g_x=g_x.view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, -1, h,w)
            if self.ssp:
                phi_x=self.p(phi_x)
            phi_x=phi_x.view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N
        # print(f_div_C.shape)
        # print(g_x.shape)
        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z

class ZeroInitBN(nn.BatchNorm3d):
    """BatchNorm with zero initialization."""

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.zeros_(self.weight)
            nn.init.zeros_(self.bias)

class Nonlocal(nn.Module):
    """Lightweight Non-Local Module.
    """
    def __init__(self, n_feature, nl_c, nl_cs, nl_s, batch_norm_kwargs=None):
        super(Nonlocal, self).__init__()
        self.n_feature = n_feature
        self.nl_c = nl_c
        self.nl_cs = nl_cs
        self.nl_s = nl_s
        self.depthwise_conv = nn.Conv3d(n_feature,
                                        n_feature,
                                        (1,3,3),
                                        1, padding=(0,(3 - 1) // 2,(3 - 1) // 2),
                                        groups=n_feature,
                                        bias=False)

        if batch_norm_kwargs is None:
            batch_norm_kwargs = {}

        self.bn = nn.BatchNorm3d(n_feature)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, l):
        N, n_in, C, H, W = list(l.shape)
        reduced_CHW = (C // self.nl_cs) * (H // self.nl_s) * (W // self.nl_s)
        l_reduced = l[:, :, ::self.nl_cs, ::self.nl_s, ::self.nl_s]
        theta = l[:, :int(self.nl_c * n_in), :, :, :]
        phi = l_reduced[:, :int(self.nl_c * n_in), :, :, :]
        g = l_reduced
        f = torch.einsum('nichw,njchw->nij', phi, g)
        f = torch.einsum('nij,nichw->njchw', f, theta)
        f = f /(C * H * W)
        f = self.act(self.bn(self.depthwise_conv(f)))
        return f + l

    def __repr__(self):
        return '{}({}, nl_c={}, nl_s={}'.format(self._get_name(),self.n_feature, self.nl_c,self.nl_s)


