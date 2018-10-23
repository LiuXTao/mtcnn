
import torch
from nets.models import PNet, RNet, ONet

def create_mtcnn_net(pnet_path=None, rnet_path=None, onet_path=None, use_cuda=False):
    pnet, rnet, onet = None, None, None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if pnet_path is not None:
        pnet = PNet(use_cuda=use_cuda)
        pnet.load_state_dict(torch.load(pnet_path))

        if use_cuda:
            pnet.to(device)

        pnet.eval()

    if rnet_path is not None:
        rnet = RNet(use_cuda=use_cuda)
        rnet.load_state_dict(torch.load(rnet_path))
        if use_cuda:
            rnet.to(device)
        rnet.eval()

    if onet_path is not None:
        onet = ONet(use_cuda=use_cuda)
        onet.load_state_dict(torch.load(onet_path))
        if use_cuda:
            onet.to(device)

    return pnet, rnet, onet
