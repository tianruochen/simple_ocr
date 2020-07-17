from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch

from .networks.msra_resnet import get_pose_net
from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn

_model_factory = {
    'res': get_pose_net,  # default Resnet with deconv
    'resdcn': get_pose_net_dcn,
}

# opt.arch = "res_18"
        # opt.heads = opt.heads = {'hm': 1,
        #                          'wh': 2, "reg": 2}
        # opt.head_conv = 64
def create_model(arch, heads, head_conv):
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0    #18
    arch = arch[:arch.find('_')] if '_' in arch else arch    #res
    get_model = _model_factory[arch]
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
    return model

# model = resnet18
# model_path = '/data1/zhaoshiyu/temp/model_files/ocr_v1024/model_last.pth'
def load_model(model, model_path):

    # 把所有的张量加载到CPU中
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    return model


if __name__ == "__main__":
    print("sth")

