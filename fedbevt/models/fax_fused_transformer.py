import torch.nn as nn
from einops import rearrange
from fedbevt.models.sub_modules.fax_modules import FAXModule
from fedbevt.models.backbones.resnet_ms import ResnetEncoder
from fedbevt.models.sub_modules.naive_decoder import NaiveDecoder
from fedbevt.models.sub_modules.bev_seg_head import BevSegHead


class FaxFusedTransformer(nn.Module):
    def __init__(self, config):
        super(FaxFusedTransformer, self).__init__()
        # encoder params
        self.encoder = ResnetEncoder(config['encoder'])

        # cvm params
        cvm_params = config['fax']
        cvm_params['backbone_output_shape'] = self.encoder.output_shapes
        self.fax = FAXModule(cvm_params)

        # decoder params
        decoder_params = config['decoder']
        # decoder for dynamic and static differet
        self.decoder = NaiveDecoder(decoder_params)

        self.target = config['target']
        self.seg_head = BevSegHead(self.target,
                                   config['seg_head_dim'],
                                   config['output_class'])

    def forward(self, batch_dict):
        x = batch_dict['inputs']
        b, l, m, _, _, _ = x.shape

        x = self.encoder(x)
        batch_dict.update({'features': x})
        x = self.fax(batch_dict)

        # dynamic head
        x = self.decoder(x)
        x = rearrange(x, 'b l c h w -> (b l) c h w')

        output_dict = self.seg_head(x, b, l)

        return output_dict