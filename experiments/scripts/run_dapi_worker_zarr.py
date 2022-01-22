import glob as glob
import os
import numpy as np
import sys
from tqdm import tqdm
import argparse
import json
import zarr

from dapi.utils import open_zarr_image, save_zarr_image, get_image_pairs
from dapi.mask import get_mask
from dapi.attribute import get_attribution

parser = argparse.ArgumentParser()
parser.add_argument('--worker', type=int, required=True)
parser.add_argument('--id_min', type=int, required=True)
parser.add_argument('--id_max', type=int, required=True)
# parser.add_argument('--img_dir', type=str, required=True)
parser.add_argument('--zarr_real',type=str, required=True, help='path to zarr store containing the real images')
parser.add_argument('--zarr_real_array',type=str,required=False,default=None,help='name of array containing real images in zarr store')
parser.add_argument('--zarr_fake',type=str, required=True, help='path to zarr store containing the fake images')
parser.add_argument('--zarr_fake_array',type=str,required=False,default=None,help='name of array containing fake images in zarr store')
parser.add_argument('--real_class', type=int, required=True)
parser.add_argument('--fake_class', type=int, required=True)
parser.add_argument('--class_names', nargs="+", required=True)
parser.add_argument('--net_module', type=str, required=True)
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--input_shape', nargs="+", type=int, required=True)
parser.add_argument('--output_classes', type=int, required=True)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--abs_attr', action="store_true")
parser.add_argument('--methods', nargs="+", default=["ig", "grads", "dl", "gc", "ggc", "ingrad", "random", "residual"])
parser.add_argument('--downsample_factors', nargs="+")
parser.add_argument('--bidirectional', type=int, required=True)
parser.add_argument('--max_images', type=int, required=False, default=None)
parser.add_argument('--channels', type=int, required=True)
parser.add_argument('--fmaps',type=int,default=None)

def run_worker(worker,
        id_min,
        id_max,
        # img_dir,
        zarr_real,
        zarr_real_array,
        zarr_fake,
        zarr_fake_array,
        real_class,
        fake_class,
        net_module,
        checkpoint,
        input_shape,
        output_classes,
        out_dir,
        abs_attr,
        methods,
        class_names,
        downsample_factors,
        bidirectional,
        max_images,
        channels,
        fmaps,
        write_opt=True):

    bidirectional = bool(bidirectional)
    real_class_name = class_names[real_class]
    fake_class_name = class_names[fake_class]
    zarr_out = f'{out_dir}/{real_class_name}_{fake_class_name}.opt_images.zarr'
    # reals = []
    # fakes = []
    # for real_class, fake_class in zip(real_classes, fake_classes):
        # image_pairs = get_image_pairs(img_dir, real_class, fake_class)
        # reals_dir = [(p[0],class_names.index(str(real_class))) for p in image_pairs]
        # fakes_dir = [(p[1],class_names.index(str(fake_class))) for p in image_pairs]

        # if max_images is not None:
        #     reals.extend(reals_dir[:max_images])
        #     fakes.extend(fakes_dir[:max_images])
        # else:
        #     reals.extend(reals_dir)
        #     fakes.extend(fakes_dir)

    for i in tqdm(range(id_min, id_max), position=worker):
        if i>max_images:
            break
        # real = reals[i]
        # fake = fakes[i]
        # with HiddenPrints():
            # img_idx = i
            # img_dir = os.path.join(out_dir, f"{i}")

            # real_img = real[0]
            # fake_img = fake[0]
            # real_class = real[1]
            # fake_class = fake[1]

        real_img = open_zarr_image(zarr_real, i, input_shape=input_shape)#, array_name=zarr_real_array)
        fake_img = open_zarr_image(zarr_fake, i, input_shape=input_shape)#, array_name=zarr_fake_array)

        if methods is None:
            attrs, attrs_names = get_attribution(real_img, fake_img, real_class,
                                                    fake_class, net_module, checkpoint,
                                                    input_shape, channels, fmaps,
                                                    output_classes=output_classes,
                                                    bidirectional=bidirectional,
                                                    downsample_factors=downsample_factors)
        else:
            attrs, attrs_names = get_attribution(real_img, fake_img, real_class,
                                                    fake_class, net_module, checkpoint,
                                                    input_shape, channels, fmaps, methods,
                                                    output_classes=output_classes,
                                                    bidirectional=bidirectional,
                                                    downsample_factors=downsample_factors)


        for attr, name in zip(attrs, attrs_names):
            if abs_attr:
                attr = np.abs(attr)

            result_dict, img_names, imgs_all = get_mask(attr, real_img, fake_img,
                                                        real_class, fake_class,
                                                        net_module, checkpoint, input_shape,
                                                        channels, fmaps, output_classes=output_classes,
                                                        downsample_factors=downsample_factors)




            method_dir = os.path.join(out_dir, name)
            if not os.path.exists(method_dir):
                os.makedirs(method_dir)

                with open(os.path.join(method_dir, f"{real_class_name}_to_{fake_class_name}_results.txt"), 'a') as f:
                    print(result_dict, file=f)

            if write_opt:
                thr_idx, thr, mask_size, mask_score = get_optimal_mask(result_dict, input_shape[0])
                imgs_opt = imgs_all[thr_idx]
                
                for img_opt, img_name in zip(imgs_opt, img_names):
                    # out_path = os.path.join(imgs_dir, f"{img_name}.png")
                    print(f'img_name: {img_name}, img shape: {img_opt.shape}, img max: {img_opt.max()}')
                    save_zarr_image(img_opt, zarr_out, array_name=f'{name}/{img_name}', renorm=True)

                z_out = zarr.open(zarr_out)
                z_out.attrs['metadata'] = {"real_class": real_class,
                                "fake_class": fake_class,
                                "thr": thr,
                                "mask_size": mask_size,
                                "mask_score": mask_score,
                                "thr_idx": int(thr_idx),
                                "image_idx":i
                                }

                # with open(f'{imgs_dir}/img_info.json', "w+") as f:
                #     json.dump({"real_class": real_class,
                #                "fake_class": fake_class,
                #                "thr": thr,
                #                "mask_size": mask_size,
                #                "mask_score": mask_score,
                #                "thr_idx": int(thr_idx)}, f)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def parse_args(args):
    arg_dict = vars(args)
    down = arg_dict["downsample_factors"]
    if down[0] != "None":
        down = [int(k) for k in arg_dict["downsample_factors"]]
        down = [(down[i], down[i+1]) for i in range(0,len(down),2)]
    else:
        down = None
    arg_dict["downsample_factors"] = down
    return arg_dict

def get_optimal_mask(result_dict, size):
    def ascore(m_s, m_n):
        return m_n**2 + (1 - m_s)**2

    ascores = []
    thrs = []
    mask_sizes = []
    mask_scores = []
    for thr, m in result_dict.items():
        mask_score = m[0]
        mask_size = m[1]/float(size)**2
        ascores.append(ascore(mask_score, mask_size))
        thrs.append(thr)
        mask_sizes.append(mask_size)
        mask_scores.append(mask_score)

    thr_idx = np.argmin(ascores)
    thr = thrs[thr_idx]
    mask_size = mask_sizes[thr_idx]
    mask_score = mask_scores[thr_idx]

    return thr_idx, thr, mask_size, mask_score

if __name__ == "__main__":
    args = parser.parse_args()
    arg_dict = parse_args(args)
    run_worker(**arg_dict)
