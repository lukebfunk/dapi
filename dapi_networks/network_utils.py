import importlib
import torch
import torch.nn.functional as F

from dapi.utils import image_to_tensor

def init_network(checkpoint_path=None, input_shape=(128,128), net_module="Vgg2D",
                 input_nc=1, fmaps=None, output_classes=6, gpu_ids=[], eval_net=True, require_grad=False,
                 downsample_factors=None):
    """
    checkpoint_path: Path to train checkpoint to restore weights from

    input_nc: input_channels for aux net

    aux_net: name of aux net
    """
    if net_module=="gemelli":
        from funlib.learn.torch import models
        net = models.Vgg2D(input_size=input_shape, input_fmaps=input_nc, output_classes=output_classes,
                        downsample_factors=downsample_factors,fmaps=fmaps)
    else:
        net_mod = importlib.import_module(f"dapi_networks.{net_module}")
        net_class = getattr(net_mod, f'{net_module}')
        if net_module == "Vgg2D":
            net = net_class(input_size=input_shape, input_channels=input_nc, output_classes=output_classes,
                            downsample_factors=downsample_factors)
        else:
            net = net_class(input_size=input_shape, input_channels=input_nc, output_classes=output_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    if eval_net:
        net.eval()

    if require_grad:
        for param in net.parameters():
            param.requires_grad = True
    else:
        for param in net.parameters():
            param.requires_grad = False

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        try:
            net.load_state_dict(checkpoint['model_state_dict'])
        except KeyError:
            try: 
                net.load_state_dict(checkpoint['state_dict'])
            except RuntimeError:
                try:
                    net.load_state_dict({k.lstrip('model.'):v for k,v in checkpoint['state_dict'].items()})
                except:
                    net.load_state_dict(checkpoint)
    return net

def run_inference(net, im):
    """
    Net: network object
    input_image: Normalized 2D input image.
    """
    im_tensor = image_to_tensor(im)
    class_probs = F.softmax(net(im_tensor), dim=1)
    return class_probs
