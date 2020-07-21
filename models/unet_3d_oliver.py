import torch

#%% Building blocks
def conv_block_3d(in_dim,out_dim,act_fn):
    model = torch.nn.Sequential(
        torch.nn.Conv3d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model

def conv_trans_block_3d(in_dim,out_dim,act_fn):
    model = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        torch.nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model

def maxpool_3d():
    pool = torch.nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
    return pool

def conv_block_2_3d(in_dim,out_dim,act_fn):
    model = torch.nn.Sequential(
        conv_block_3d(in_dim,out_dim,act_fn),
        torch.nn.Conv3d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm3d(out_dim),
    )
    return model


#def conv_block_3_3d(in_dim,out_dim,act_fn):
#    model = torch.nn.Sequential(
#        conv_block_3d(in_dim,out_dim,act_fn),
#        conv_block_3d(out_dim,out_dim,act_fn),
#        torch.nn.Conv3d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
#        torch.nn.BatchNorm3d(out_dim),
#    )
#    return model

#%% Plain vanilla 3D Unet
class Unet3D(torch.nn.Module):
    def __init__(self,in_dim=1,out_dim=1,num_filter=2):
        super(Unet3D,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        act_fn = torch.nn.LeakyReLU(0.2, inplace=True)

        self.down_1 = conv_block_2_3d(self.in_dim,self.num_filter,act_fn)
        self.pool_1 = maxpool_3d()
        self.down_2 = conv_block_2_3d(self.num_filter,self.num_filter*2,act_fn)
        self.pool_2 = maxpool_3d()
        self.down_3 = conv_block_2_3d(self.num_filter*2,self.num_filter*4,act_fn)
        self.pool_3 = maxpool_3d()

        self.bridge = conv_block_2_3d(self.num_filter*4,self.num_filter*8,act_fn)

        self.trans_1 = conv_trans_block_3d(self.num_filter*8,self.num_filter*8,act_fn)
        self.up_1    = conv_block_2_3d(self.num_filter*12,self.num_filter*4,act_fn)
        self.trans_2 = conv_trans_block_3d(self.num_filter*4,self.num_filter*4,act_fn)
        self.up_2    = conv_block_2_3d(self.num_filter*6,self.num_filter*2,act_fn)
        self.trans_3 = conv_trans_block_3d(self.num_filter*2,self.num_filter*2,act_fn)
        self.up_3    = conv_block_2_3d(self.num_filter*3,self.num_filter*1,act_fn)

        self.out = conv_block_3d(self.num_filter,out_dim,act_fn)

    def forward(self,x):
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        bridge = self.bridge(pool_3)

        trans_1  = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1,down_3],dim=1)
        up_1     = self.up_1(concat_1)
        trans_2  = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2,down_2],dim=1)
        up_2     = self.up_2(concat_2)
        trans_3  = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3,down_1],dim=1)
        up_3     = self.up_3(concat_3)

        out = self.out(up_3)
        out = out[:,:,2:-2,2:-2,2:-2]
        out = torch.sigmoid(out)
        return out

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        t_ = torch.load(path)
        print("T_ {}".format(t_))
        self.load_state_dict(t_)
        print("Loaded model from {0}".format(str(path)))

    def weight_hist(self, x):
        pass

#%% Bit more complex 3D Unet (4 instead of 3 layers, 192 instead of 48 feature channels)
class Unet3D_test(torch.nn.Module):

    def __init__(self,in_dim,out_dim,num_filter):
        super(Unet3D_test,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        act_fn = torch.nn.LeakyReLU(0.2, inplace=True)

        self.down_1 = conv_block_2_3d(self.in_dim,self.num_filter,act_fn)
        self.pool_1 = maxpool_3d()
        self.down_2 = conv_block_2_3d(self.num_filter,self.num_filter*2,act_fn)
        self.pool_2 = maxpool_3d()
        self.down_3 = conv_block_2_3d(self.num_filter*2,self.num_filter*4,act_fn)
        self.pool_3 = maxpool_3d()
        self.down_4 = conv_block_2_3d(self.num_filter*4,self.num_filter*8,act_fn)
        self.pool_4 = maxpool_3d()

        self.bridge = conv_block_2_3d(self.num_filter*8,self.num_filter*16,act_fn)

        self.trans_0 = conv_trans_block_3d(self.num_filter*16,self.num_filter*16,act_fn)
        self.up_0    = conv_block_2_3d(self.num_filter*24,self.num_filter*8,act_fn)
        self.trans_1 = conv_trans_block_3d(self.num_filter*8,self.num_filter*8,act_fn)
        self.up_1    = conv_block_2_3d(self.num_filter*12,self.num_filter*4,act_fn)
        self.trans_2 = conv_trans_block_3d(self.num_filter*4,self.num_filter*4,act_fn)
        self.up_2    = conv_block_2_3d(self.num_filter*6,self.num_filter*2,act_fn)
        self.trans_3 = conv_trans_block_3d(self.num_filter*2,self.num_filter*2,act_fn)
        self.up_3    = conv_block_2_3d(self.num_filter*3,self.num_filter*1,act_fn)

        self.out = conv_block_3d(self.num_filter,out_dim,act_fn)

    def forward(self,x):
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        trans_0  = self.trans_0(bridge)
        concat_0 = torch.cat([trans_0,down_4],dim=1)
        up_0     = self.up_0(concat_0)
        trans_1  = self.trans_1(up_0)
        concat_1 = torch.cat([trans_1,down_3],dim=1)
        up_1     = self.up_1(concat_1)
        trans_2  = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2,down_2],dim=1)
        up_2     = self.up_2(concat_2)
        trans_3  = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3,down_1],dim=1)
        up_3     = self.up_3(concat_3)

        out = self.out(up_3)

        return out


#%% 3D Unet with same # of layer & feature channels as 2Dunet768
# NOTE: this would only workl with cubes of, e.g., 192 pixel edge length due to 6 downpooling operations
# In this case, the spatial dimension would shrink to 3
# Since this can only be hardly produced with our data and would never ever fit into the memory, we adapted to Unet3D_adapted
class Unet3D768(torch.nn.Module):

    def __init__(self,in_dim,out_dim):

        super(Unet3D768,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = torch.nn.LeakyReLU(0.2, inplace=True)

        self.down_1 = conv_block_2_3d(self.in_dim,24,act_fn)
        self.pool_1 = maxpool_3d()
        self.down_2 = conv_block_2_3d(24,64,act_fn)
        self.pool_2 = maxpool_3d()
        self.down_3 = conv_block_2_3d(64,128,act_fn)
        self.pool_3 = maxpool_3d()
        self.down_4 = conv_block_2_3d(128,256,act_fn)
        self.pool_4 = maxpool_3d()
        self.down_5 = conv_block_2_3d(256,512,act_fn)
        self.pool_5 = maxpool_3d()
        self.down_6 = conv_block_2_3d(512,768,act_fn)
        self.pool_6 = maxpool_3d()

        self.bridge = conv_block_2_3d(768,768,act_fn) # before, the output was doubled!

        self.trans_6 = conv_trans_block_3d(768,768,act_fn)
        self.up_6    = conv_block_2_3d(768+512,512,act_fn)
        self.trans_5 = conv_trans_block_3d(512,512,act_fn)
        self.up_5    = conv_block_2_3d(512+256,256,act_fn)
        self.trans_4 = conv_trans_block_3d(256,256,act_fn)
        self.up_4    = conv_block_2_3d(256+128,128,act_fn)
        self.trans_3 = conv_trans_block_3d(128,128,act_fn)
        self.up_3    = conv_block_2_3d(128+64,64,act_fn)
        self.trans_2 = conv_trans_block_3d(64,64,act_fn)
        self.up_2    = conv_block_2_3d(64+24,24,act_fn)
        self.trans_1 = conv_trans_block_3d(24,24,act_fn)
        self.up_1    = conv_block_2_3d(24+24,24,act_fn)

        self.out = conv_block_3d(24,out_dim,act_fn)


    def forward(self,x):
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)
        down_5 = self.down_5(pool_4)
        pool_5 = self.pool_5(down_5)
        down_6 = self.down_6(pool_5)
        pool_6 = self.pool_6(down_6)

        bridge = self.bridge(pool_6)

        trans_6  = self.trans_6(bridge)
        up_6     = self.up_6(torch.cat([trans_6,down_6],dim=1))
        trans_5  = self.trans_5(up_6)
        up_5     = self.up_5(torch.cat([trans_5,down_5],dim=1))
        trans_4  = self.trans_4(up_5)
        up_4     = self.up_4(torch.cat([trans_4,down_4],dim=1))
        trans_3  = self.trans_3(up_4)
        up_3     = self.up_3(torch.cat([trans_3,down_3],dim=1))
        trans_2  = self.trans_2(up_3)
        up_2     = self.up_2(torch.cat([trans_2,down_2],dim=1))
        trans_1  = self.trans_1(up_2)
        up_1     = self.up_1(torch.cat([trans_1,down_1],dim=1))

        out = self.out(up_1)

        return out


#%% 3D Unet with ALMOST the same # of layer & feature channels as 2Dunet768
# We have 1 layer less but scale up feature channels more quickly
class Unet3D_adapted(torch.nn.Module):

    def __init__(self,in_dim,out_dim):

        super(Unet3D_adapted,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = torch.nn.LeakyReLU(0.2, inplace=True)

        self.down_1 = conv_block_2_3d(self.in_dim,24,act_fn)
        self.pool_1 = maxpool_3d()
        self.down_2 = conv_block_2_3d(24,64,act_fn)
        self.pool_2 = maxpool_3d()
        self.down_3 = conv_block_2_3d(64,128,act_fn)
        self.pool_3 = maxpool_3d()
        self.down_4 = conv_block_2_3d(128,256,act_fn)
        self.pool_4 = maxpool_3d()
        self.down_5 = conv_block_2_3d(256,512,act_fn)
        self.pool_5 = maxpool_3d()

        self.bridge = conv_block_2_3d(512,512,act_fn) # before, the output was doubled!

        self.trans_5 = conv_trans_block_3d(512,512,act_fn)
        self.up_5    = conv_block_2_3d(512+512,256,act_fn)
        self.trans_4 = conv_trans_block_3d(256,256,act_fn)
        self.up_4    = conv_block_2_3d(256+256,128,act_fn)
        self.trans_3 = conv_trans_block_3d(128,128,act_fn)
        self.up_3    = conv_block_2_3d(128+128,64,act_fn)
        self.trans_2 = conv_trans_block_3d(64,64,act_fn)
        self.up_2    = conv_block_2_3d(64+64,24,act_fn)
        self.trans_1 = conv_trans_block_3d(24,24,act_fn)
        self.up_1    = conv_block_2_3d(24+24,24,act_fn)

        self.out = conv_block_3d(24,out_dim,act_fn)


    def forward(self,x):
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)
        down_5 = self.down_5(pool_4)
        pool_5 = self.pool_5(down_5)

        bridge = self.bridge(pool_5)

        trans_5  = self.trans_5(bridge)
        up_5     = self.up_5(torch.cat([trans_5,down_5],dim=1))
        trans_4  = self.trans_4(up_5)
        up_4     = self.up_4(torch.cat([trans_4,down_4],dim=1))
        trans_3  = self.trans_3(up_4)
        up_3     = self.up_3(torch.cat([trans_3,down_3],dim=1))
        trans_2  = self.trans_2(up_3)
        up_2     = self.up_2(torch.cat([trans_2,down_2],dim=1))
        trans_1  = self.trans_1(up_2)
        up_1     = self.up_1(torch.cat([trans_1,down_1],dim=1))

        out = self.out(up_1)
        return out
