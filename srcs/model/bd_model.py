from srcs.model.bd_modules import Conv, ResBlock, MLP, CUnet
import torch.nn as nn
import torch
from srcs.model.bd_utils import PositionalEncoding

class BDNeRV_RC(nn.Module):
    # recursive frame reconstruction
    def __init__(self):
        super(BDNeRV_RC, self).__init__()
        # params
        n_colors = 3
        # -- n_resblock = 4
        n_resblock = 0
        # -- n_feats = 32
        n_feats = 8
        kernel_size = 3
        padding = 1

        pos_b, pos_l = 1.25, 80  # position encoding params
        mlp_dim_list = [2*pos_l, 512, n_feats*4*2] # (160, 512, 256)
        mlp_act = 'gelu'

        # main body
        self.mainbody = CUnet(n_feats=n_feats, n_resblock=n_resblock,
                              kernel_size=kernel_size, padding=padding)

        # output block
        """
        OutBlock = [ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
                    ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
                    Conv(input_channels=n_feats, n_feats=n_colors, kernel_size=kernel_size, padding=padding)]
        """
        OutBlock = [ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding),
                    Conv(input_channels=n_feats, n_feats=n_colors, kernel_size=kernel_size, padding=padding)]
        self.out = nn.Sequential(*OutBlock)

        # feature block
        # -- The structure of a ResBlock:
        #        X
        #        |_______
        #        |       |
        #    +---V---+   |
        #    |  CONV |   |
        #    +---V---+   |
        #        |       |
        #    +---V---+   |
        #    |  ReLU |   |
        #    +---V---+   |
        #        |       |
        #    +---V---+   |
        #    |  CONV |   |
        #    +---V---+   |
        #        |       |
        #    +---V---+   |
        #    |   +   <---/
        #    +---V---+   
        #        |       
        #        V
        # -- The numbers of input and output channels are both n_feats=32 in the original code.
        """
        FeatureBlock = [Conv(input_channels=n_colors, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        """
        FeatureBlock = [Conv(input_channels=n_colors, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.feature = nn.Sequential(*FeatureBlock)

        # concatenation fusion block
        """
        CatFusion = [Conv(input_channels=n_feats*2, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats,
                                 kernel_size=kernel_size, padding=padding),
                        ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        """
        CatFusion = [Conv(input_channels=n_feats*2, n_feats=n_feats, kernel_size=kernel_size, padding=padding, act=True),
                     ResBlock(Conv, n_feats=n_feats, kernel_size=kernel_size, padding=padding)]
        self.catfusion = nn.Sequential(*CatFusion)

        # -- Positional encoding maps the frame index to high-dimensional embedding space, enhancing the network's capacity 
        #    in fitting data with high-frequency variations
        #    ** Tancik et al. (2020) "Fourier features let networks learn high frequency functions in low dimensional domains"
        #       Advances in Neural Information Processing Systems, Curran Associates, Inc., vol 33, pp 75377547
        #    ** Mildenhall et al. (2020) "NeRF: Representing scenes as neural radiance fields for view synthesis" Computer Vision
        #       - ECCV 2020, Springer International Publishing, pp 405-421.
        #    ** Li et al. (2020) "E-NeRF: Expedite neural video representation with disentangled spatial-temporal context" Computer
        #       Vision - ECCV 2022, Springer Nature Switzerland, pp 267-284.
        #
        # -- The embeddings are just a list/tensor of length, L, defined as:
        # 
        #     [sin((b0**0)*math.pi*ti), cos((b0**0)*math.pi*ti),
        #      sin((b0**1)*math.pi*ti), cos((b0**1)*math.pi*ti),
        #      sin((b0**2)*math.pi*ti), cos((b0**2)*math.pi*ti),
        #                     ... ... ... ...
        #      sin((b0**(L-1))*math.pi*ti), cos((b0**(L-1))*math.pi*ti)]
        #      
        # -- pos_b = 1.25 (this is b0)
        # -- pos_l = 80   (this is L)
        # 
        # -- Both parameters are empiricially determined.
        self.pe_t = PositionalEncoding(pe_embed_b=pos_b, pe_embed_l=pos_l)

        # mlp
        self.embed_mlp = MLP(dim_list=mlp_dim_list, act=mlp_act)

    def forward(self, ce_blur, time_idx, ce_code):
        """
        Returns deblurred images.
        Args:
            ce_blur:
                -- An input blurry image.
                -- In the original code, the size is (1, 3, 720, 1280)
            time_idx:
                -- The frame index.
                -- It's provided by another module called CEBlurNet that performs coded-exposure.
                -- Its shape is [n_frame=8, 1] in the original code.
                -- It's a column vector containing evenly spaced values between 0 and 1.
            ce_code:80
                -- The mask used to perform coded exposure.
                -- It's also provided by CEBlurNet.
                -- Its shape is also [n_frame=8, 1] in the original code.
        """

        # time index: [frame_num,1]
        # t_embed
        # -- 2*code-1 is to transform the binary code into an element of {-1, 1}
        # -- Each element of the mask is represented by a vector of 160 elements.
        # -- If the mask is 1, the embeddings are just the 160 elements returned by the positional encoder.
        # -- If the mask is 0, the embeddings are multiplication of -1 and the 160 elements returned by the positional encoder.
        # -- t_pe_ is a list of length n_frame=8, each of the elements of the list is a vector of length 2*pos_l=160
        t_pe_ = [self.pe_t(idx)*(2*code-1)
                 for idx, code in zip(time_idx, ce_code)]  # [frame_num*[pos_l*2,1]]

        t_pe = torch.cat(t_pe_, dim=0)  # [frame_num, pos_l*2]
        t_embed = self.embed_mlp(t_pe)  # [frame_num, n_feats*4*2]
        # t_manip = self.manip_mlp(t_pe)

        # ce_blur feature
        ce_feature = self.feature(ce_blur)  # [b, c, h, w]

        # main body
        output_list = []
        for k in range(len(time_idx)):
            if k==0:
                main_feature = ce_feature
            else:
                # since k=2, cat pre-feature with ce_feature as input feature
                cat_feature = torch.cat((feat_out_k, ce_feature),dim=1)
                main_feature = self.catfusion(cat_feature)
            feat_out_k = self.mainbody(main_feature, t_embed[k])
            output_k = self.out(feat_out_k)
            output_list.append(output_k)

        output = torch.stack(output_list, dim=1)

        return output
