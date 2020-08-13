import torch

#%% Building blocks
def conv_block_2d(in_dim,out_dim,act_fn):
    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model

def conv_trans_block_2d(in_dim,out_dim,act_fn):
    model = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        torch.nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model

def maxpool_2d():
    pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool

def conv_block_2_2d(in_dim,out_dim,act_fn):
    model = torch.nn.Sequential(
        conv_block_2d(in_dim,out_dim,act_fn),
        torch.nn.Conv2d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(out_dim),
    )
    return model

#%% Plain vanilla 3D Unet
class Unet2D(torch.nn.Module):
    def __init__(self,in_dim=1,out_dim=1,num_filter=2):
        super(Unet2D,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        act_fn = torch.nn.LeakyReLU(0.2, inplace=True)

        self.down_1 = conv_block_2_2d(self.in_dim,self.num_filter,act_fn)
        self.pool_1 = maxpool_2d()
        self.down_2 = conv_block_2_2d(self.num_filter,self.num_filter*2,act_fn)
        self.pool_2 = maxpool_2d()
        self.down_3 = conv_block_2_2d(self.num_filter*2,self.num_filter*4,act_fn)
        self.pool_3 = maxpool_2d()

        self.bridge = conv_block_2_2d(self.num_filter*4,self.num_filter*8,act_fn)

        self.trans_1 = conv_trans_block_2d(self.num_filter*8,self.num_filter*8,act_fn)
        self.up_1    = conv_block_2_2d(self.num_filter*12,self.num_filter*4,act_fn)
        self.trans_2 = conv_trans_block_2d(self.num_filter*4,self.num_filter*4,act_fn)
        self.up_2    = conv_block_2_2d(self.num_filter*6,self.num_filter*2,act_fn)
        self.trans_3 = conv_trans_block_2d(self.num_filter*2,self.num_filter*2,act_fn)
        self.up_3    = conv_block_2_2d(self.num_filter*3,self.num_filter*1,act_fn)

        self.out = conv_block_2d(self.num_filter,out_dim,act_fn)

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
        self.load_state_dict(t_)

    def weight_hist(self, x):
        pass

