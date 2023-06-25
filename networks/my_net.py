import torch
from torch import nn


def sharpening(P):
    T = 1/0.1
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling_function(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling = 1):
        super(Upsampling_function, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        
        self.dropout = nn.Dropout3d(p=0.3, inplace=False)

    def forward(self, input, en=[]):
        
        if len(en) != 0:
            x1 = self.block_one(input)
            x1 = x1 + en[4]
            x1_dw = self.block_one_dw(x1)
    
            x2 = self.block_two(x1_dw)
            x2 = x2 + en[3]
            x2_dw = self.block_two_dw(x2)
    
            x3 = self.block_three(x2_dw)
            x3 = x3 + en[2]
            x3_dw = self.block_three_dw(x3)
    
            x4 = self.block_four(x3_dw)
            x4 = x4 + en[1]
            x4_dw = self.block_four_dw(x4)
    
            x5 = self.block_five(x4_dw)
            x5 = x5 + en[0]    # for 5% data
    
            if self.has_dropout:
                x5 = self.dropout(x5)

        else:    
            x1 = self.block_one(input)
            x1_dw = self.block_one_dw(x1)
    
            x2 = self.block_two(x1_dw)
            x2_dw = self.block_two_dw(x2)
    
            x3 = self.block_three(x2_dw)
            x3_dw = self.block_three_dw(x3)
    
            x4 = self.block_four(x3_dw)
            x4_dw = self.block_four_dw(x4)
    
            x5 = self.block_five(x4_dw)
    
            if self.has_dropout:
                x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res
    
    
class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, up_type=0):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization, mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization, mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization, mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization, mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features, f1='none', f2='none'):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]
        
        if f1 == 'none' and f2 == 'none':
            x5_up_ori = self.block_five_up(x5)
            x5_up = x5_up_ori + x4
    
            x6 = self.block_six(x5_up)
            x6_up_ori = self.block_six_up(x6)
            x6_up = x6_up_ori + x3
    
            x7 = self.block_seven(x6_up)
            x7_up_ori = self.block_seven_up(x7)
            x7_up = x7_up_ori + x2
    
            x8 = self.block_eight(x7_up)
            x8_up_ori = self.block_eight_up(x8)
            x8_up = x8_up_ori + x1
            
            x9 = self.block_nine(x8_up)
            if self.has_dropout:
                x9 = self.dropout(x9)
            out_seg = self.out_conv(x9)
            
        elif f1 != 'none' and f2 != 'none':
            m5, m4, m3, m2, m1 = f1[0], f1[1], f1[2], f1[3], f1[4]
            w5, w4, w3, w2, w1 = torch.sigmoid(m5), torch.sigmoid(m4), torch.sigmoid(m3), torch.sigmoid(m2), torch.sigmoid(m1)
            m5_, m4_, m3_, m2_, m1_ = f2[0], f2[1], f2[2], f2[3], f2[4]
            w5_, w4_, w3_, w2_, w1_ = torch.sigmoid(m5_), torch.sigmoid(m4_), torch.sigmoid(m3_), torch.sigmoid(m2_), torch.sigmoid(m1_)
            
            x5 = x5 + 0.5*(x5*w5 + x5*w5_)
            x5_up_ori = self.block_five_up(x5)
            x5_up = x5_up_ori + 0.5*(x4*w4 + x4*w4_)
    
            x6 = self.block_six(x5_up)
            x6_up_ori = self.block_six_up(x6)
            x6_up = x6_up_ori + 0.5*(x3*w3 + x3*w3_)
    
            x7 = self.block_seven(x6_up)
            x7_up_ori = self.block_seven_up(x7)
            x7_up = x7_up_ori + 0.5*(x2*w2 + x2*w2_)
    
            x8 = self.block_eight(x7_up)
            x8_up_ori = self.block_eight_up(x8)
            x8_up = x8_up_ori + 0.5*(x1*w1 + x1*w1_)
            
            x9 = self.block_nine(x8_up)
            if self.has_dropout:
                x9 = self.dropout(x9)
            out_seg = self.out_conv(x9)
            
        else:
            m5, m4, m3, m2, m1 = f1[0], f1[1], f1[2], f1[3], f1[4]
            w5, w4, w3, w2, w1 = torch.sigmoid(m5), torch.sigmoid(m4), torch.sigmoid(m3), torch.sigmoid(m2), torch.sigmoid(m1) #sharpening
            w5, w4, w3, w2, w1 = w5.detach(), w4.detach(), w3.detach(), w2.detach(), w1.detach()                  
            x5 = x5 + x5*w5
            x5_up_ori = self.block_five_up(x5)
            x5_up = x5_up_ori + x4*w4
    
            x6 = self.block_six(x5_up)
            x6_up_ori = self.block_six_up(x6)
            x6_up = x6_up_ori + x3*w3
    
            x7 = self.block_seven(x6_up)
            x7_up_ori = self.block_seven_up(x7)
            x7_up = x7_up_ori + x2*w2
    
            x8 = self.block_eight(x7_up)
            x8_up_ori = self.block_eight_up(x8)
            x8_up = x8_up_ori + x1*w1
            
            x9 = self.block_nine(x8_up)
            if self.has_dropout:
                x9 = self.dropout(x9)
            out_seg = self.out_conv(x9)   
        return out_seg, [x5, x5_up_ori, x6_up_ori, x7_up_ori, x8_up_ori]

class SideConv(nn.Module):
    def __init__(self, n_classes=2):
        super(SideConv, self).__init__()
        
        self.side5 = nn.Conv3d(256, n_classes, 1, padding=0)
        self.side4 = nn.Conv3d(128, n_classes, 1, padding=0)
        self.side3 = nn.Conv3d(64, n_classes, 1, padding=0)
        self.side2 = nn.Conv3d(32, n_classes, 1, padding=0)
        self.side1 = nn.Conv3d(16, n_classes, 1, padding=0)
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
    def forward(self, stage_feat):
        x5, x5_up, x6_up, x7_up, x8_up = stage_feat[0], stage_feat[1], stage_feat[2], stage_feat[3], stage_feat[4]
        out5 = self.side5(x5)
        out5 = self.upsamplex2(out5)
        out5 = self.upsamplex2(out5)
        out5 = self.upsamplex2(out5)
        out5 = self.upsamplex2(out5)
        
        out4 = self.side4(x5_up)
        out4 = self.upsamplex2(out4)
        out4 = self.upsamplex2(out4)
        out4 = self.upsamplex2(out4)
        
        out3 = self.side3(x6_up)
        out3 = self.upsamplex2(out3)
        out3 = self.upsamplex2(out3)
        
        out2 = self.side2(x7_up)
        out2 = self.upsamplex2(out2)
        
        out1 = self.side1(x8_up)
        return [out5, out4, out3, out2, out1]

class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(VNet, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
    
    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        return out_seg1

class MCNet3d_v1(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(MCNet3d_v1, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters,normalization,  has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        self.decoder3 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 2)
        self.sideconv1 = SideConv()
        self.sideconv2 = SideConv()
    
    def forward(self, input):
        features = self.encoder(input)
        out_seg1, stage_feat1 = self.decoder1(features)
        out_seg2, stage_feat2 = self.decoder2(features)
        out_seg3, stage_feat3 = self.decoder3(features, stage_feat1, stage_feat2)
        deep_out1 = self.sideconv1(stage_feat1)
        deep_out2 = self.sideconv2(stage_feat2)
        return out_seg1, out_seg2, out_seg3, deep_out1, deep_out2
    
class MCNet3d_v2(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(MCNet3d_v2, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        self.sideconv1 = SideConv()
    
    def forward(self, input, en):
        features = self.encoder(input, en)
        out_seg1, stage_feat1 = self.decoder1(features)
        out_seg2, stage_feat2 = self.decoder2(features, stage_feat1)
        deep_out1 = self.sideconv1(stage_feat1)
        
        return out_seg1, out_seg2, [stage_feat2, stage_feat1], deep_out1, []

if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from ptflops import get_model_complexity_info
    model = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=False)
    with torch.cuda.device(0):
      macs, params = get_model_complexity_info(model, (1, 112, 112, 80), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
      print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
      print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    with torch.cuda.device(0):
      macs, params = get_model_complexity_info(model, (1, 96, 96, 96), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
      print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
      print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    import ipdb; ipdb.set_trace()
