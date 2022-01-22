import os
import numpy as np
import sys
from subprocess import Popen
import itertools
import argparse
import configparser
import zarr

from dapi.utils import get_zarr_len
#get_zarr_len # open_image, save_image, get_image_pairs

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--net', type=str, required=True)

def run_dapi_zarr(zarr_real, zarr_real_array, zarr_fake, zarr_fake_array,
            real_class, fake_class, class_names, net_module, checkpoint,
            input_shape, output_classes, fmaps, out_dir, methods=None, num_workers=10,
            channels=1, submit_script="run_dapi_worker_zarr.py", max_images=None,
            submit_cmd="python", downsample_factors=None, bidirectional=False):

    '''Run attribution & mask extraction for image pairs in given directory

    Args:

        zarr_real: (''str'')

            Path to zarr store containing real images

        zarr_real_array: (''str'')

            Name of array within `zarr_real` containing real images (or None for root array)

        zarr_fake (''str'')

            Path to zarr store containing fake images

        zarr_fake_array: (''str'')

            Name of array within `zarr_fake` containing fake images (or None for root array)

        real_class: (''str'')

            Real class identifier

        fake_class: (''str'')

            Fake class identifier

        net_module: (''str'')

            Name of network file and class

        class_names: (''list of str'')

            List of output classes of classifier, ordered.

        checkpoint: (''str'')

            Path to classifier checkpoint

        input_shape: (''tuple of int'')

            Spatial input shape of classifier

        output_classes: (''int'')

            Number of classifier output classes

        out_dir: (''str'')

            Experiment output dir

        methods: (''list of str'')

            Attribution methods to consider, if None all are returned.

        num_workers: (''int'')

            Number of workers to use.

        channels: (''int'')

            Number of input image channels

        submit_script: (''str'')

            Path to submit script

        max_images: (''str'')

            Maximum number of images to run attr for, for each class.

        submit_cmd: (''str'')

            Script submit command, will be run in terminal and can
            be used to submit to a cluster.
    '''

    # reals = []
    # fakes = []
    # for real_class, fake_class in zip(real_classes, fake_classes):
    #     image_pairs = get_image_pairs(img_dir, real_class, fake_class)
    #     reals_dir = [(p[0],real_class) for p in image_pairs]
    #     fakes_dir = [(p[1],fake_class) for p in image_pairs]

    #     if max_images is not None:
    #         reals.extend(reals_dir[:max_images])
    #         fakes.extend(fakes_dir[:max_images])
    #     else:
    #         reals.extend(reals_dir)
    #         fakes.extend(fakes_dir)

    methods_string = None
    if methods is not None:
        methods_string = " --methods "
        for method in methods:
            methods_string += f" {method}"

    n_images = get_zarr_len(zarr_real)
    n_images_per_worker = n_images/num_workers
    n_rest = n_images % num_workers

    base_cmd = f"{submit_cmd} {submit_script}"
    real_classes_string = ""
    fake_classes_string = ""

    # for real_class, fake_class in zip(real_classes, fake_classes):
    real_classes_string += f"{real_class} "
    fake_classes_string += f"{fake_class} "

    class_name_string = ""
    for c in class_names:
        class_name_string += f"{c} "

    if downsample_factors is not None:
        downsample_factor_string = ""
        for d in downsample_factors:
            for j in d:
                downsample_factor_string += f"{j} "
    else:
        downsample_factor_string = "None"

    for worker in range(num_workers):
        rest = 0
        if worker == num_workers - 1:
            rest = n_rest

        id_min = int(worker*n_images_per_worker)
        id_max = int(id_min + n_images_per_worker + rest)

        arg_cmd = f" --worker {worker} --id_min {id_min} --id_max {id_max}"+\
                  f" --zarr_real {zarr_real} --zarr_real_array {zarr_real_array}"+\
                  f" --zarr_fake {zarr_fake} --zarr_fake_array {zarr_fake_array}"+\
                  f" --real_class {real_classes_string} --fake_class {fake_classes_string}"+\
                  f" --net_module {net_module} --checkpoint {checkpoint} --input_shape {input_shape[0]} {input_shape[1]}"+\
                  f" --worker {worker} --out_dir {out_dir} --output_classes {output_classes} --class_names {class_name_string}"+\
                  f" --downsample_factors {downsample_factor_string} --bidirectional {int(bidirectional)}"+\
                  f" --channels {channels} --fmaps {fmaps}"

        if max_images is not None:
            arg_cmd += f" --max_images {max_images}"

        if methods is not None:
            arg_cmd += methods_string

        cmd = base_cmd + arg_cmd
        Popen(cmd, shell=True)

def read_exp_config(exp_config, net):
    parser = configparser.ConfigParser()
    parser.read(exp_config)
    cfg_dict = {k:v for k, v in parser[net].items()}
    cfg_dict = parse_cfg_dict(cfg_dict)
    return cfg_dict

def parse_cfg_dict(cfg_dict):
    cfg_dict_parsed = {}
    for key, values in cfg_dict.items():
        values_parsed = values
        if values_parsed != "None":
            if key == "bidirectional":
                if values_parsed == "True":
                    values_parsed = True
                elif values_parsed == "False":
                    values_parsed = False
                else:
                    raise ValueError("Bidirectional must be bool")

            if key == "downsample_factors":
                values_parsed = values_parsed.split(";")
                values_parsed = [tuple([int(j) for j in k.split(",")]) for k in values_parsed]

            elif "," in values:
                values_parsed = values_parsed.split(",")

            if key in ["num_workers", "channels", "output_classes", "max_images"]:
                values_parsed = int(values_parsed)
        else:
            values_parsed = None

        cfg_dict_parsed[key] = values_parsed
    return cfg_dict_parsed

# def get_zarr_len(z,array_name=None):
#     z = zarr.open(z,'r')
#     if array_name is not None:
#         return len(z[array_name])
#     else:
#         return len(z)

if __name__ == "__main__":
    args = parser.parse_args()
    config = read_exp_config(args.config, args.net)
    run_dapi_zarr(**config)
