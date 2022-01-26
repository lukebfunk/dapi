import numpy as np
import cv2
import copy

from dapi.utils import normalize_image, save_image
from dapi_networks import run_inference, init_network

def get_mask(attribution,
            real_img,
            fake_img, 
            real_class,
            fake_class,
            net_module,
            checkpoint_path,
            input_shape,
            input_nc,
            fmaps,
            output_classes,
            downsample_factors=None,
            sigma=11,
            struc=10,
            channel_wise=False,
            normalize=True) -> tuple[dict[float,list[float]],list[str],list[list[np.ndarray]]]:
    """
    attribution: 2D array <= 1 indicating pixel importance
    """
    # initialize
    net = init_network(checkpoint_path, input_shape, net_module, input_nc, fmaps, eval_net=True, require_grad=False, output_classes=output_classes,
                       downsample_factors=downsample_factors)
    result_dict = {}
    img_names = ["attr", "real", "fake", "hybrid", "mask_real", "mask_fake", "mask_residual", "mask_weight"]
    imgs_all = []

    # evaluate thresholds
    a_min = -1
    a_max = 1
    steps = 200
    a_range = a_max - a_min
    step = a_range/float(steps)
    for k in range(0,steps+1):
        thr = a_min + k * step

        # This is inefficient, can be fixed when using nD smoothing functions.
        mask_weight_full = np.zeros(np.shape(real_img))
        copyto_full = np.zeros(np.shape(real_img))
        copied_canvas_full = np.zeros(np.shape(real_img))
        copied_canvas_to_full = np.zeros(np.shape(real_img))

        mask_size = 0
        for c in range(input_nc):
            copyfrom = copy.deepcopy(real_img[c,:,:])
            copyto = copy.deepcopy(fake_img[c,:,:])
            copyto_ref = copy.deepcopy(fake_img[c,:,:])
            copied_canvas = np.zeros(np.shape(copyfrom))
            if channel_wise:
                mask = np.array(attribution[c,:,:] > thr, dtype=np.uint8)
            else:
                mask = np.array(np.any(attribution > thr, axis=0), dtype=np.uint8)

            # make mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(struc,struc))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            # Mask size should be counted individually for each channel.
            mask_size += np.sum(mask)
            mask_cp = copy.deepcopy(mask)

            # copy in masked patches of the real image
            mask_weight = cv2.GaussianBlur(mask_cp.astype(np.float), (sigma,sigma),0)
            copyto = np.array((copyto * (1 - mask_weight)) + (copyfrom * mask_weight), dtype=np.float)

            copied_canvas += np.array(mask_weight*copyfrom)
            copied_canvas_to = np.zeros(np.shape(copyfrom))
            copied_canvas_to += np.array(mask_weight*copyto_ref)

            mask_weight_full[c,:,:] = mask_weight
            copyto_full[c,:,:] = copyto
            copied_canvas_full[c,:,:] = copied_canvas
            copied_canvas_to_full[c,:,:] = copied_canvas_to

        mask_weight = mask_weight_full # blurred mask (per channel if applicable)
        copyto = copyto_full # fake image with masked real image patched in
        copied_canvas = copied_canvas_full # masked region from real image
        copied_canvas_to = copied_canvas_to_full # masked region from fake image

        diff_copied = copied_canvas - copied_canvas_to # difference of images in masked region

        if normalize: # if images are already normalized to the scale used by the net
            fake_img_norm = normalize_image(copy.deepcopy(fake_img))
        else:
            fake_img_norm = copy.deepcopy(fake_img)
        out_fake = run_inference(net, fake_img_norm)

        if normalize:
            real_img_norm = normalize_image(copy.deepcopy(real_img))
        else:
            real_img_norm = copy.deepcopy(real_img)
        out_real = run_inference(net, real_img_norm)

        if normalize:
            im_copied_norm = normalize_image(copy.deepcopy(copyto))
        else:
            im_copied_norm = copy.deepcopy(copyto)
        out_copyto = run_inference(net, im_copied_norm.astype(np.float32))

        imgs = [attribution, real_img_norm, fake_img_norm, im_copied_norm]
        
        if normalize:
            imgs += [normalize_image(copied_canvas),
                    normalize_image(copied_canvas_to), 
                    normalize_image(diff_copied), mask_weight]
        else:
            imgs += [copied_canvas,copied_canvas_to,diff_copied,mask_weight]

        imgs_all.append(imgs)

        mrf_score = out_copyto[0][real_class] - out_fake[0][real_class]
        result_dict[thr] = [float(mrf_score.detach().cpu().numpy()), mask_size]

    return (result_dict, # dict: threshold -> [difference in prediction, mask size]
            img_names, # list of image result type names
            imgs_all) # list of [for each threshold: list of images corresponding to img_names]
