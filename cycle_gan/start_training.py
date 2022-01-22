import subprocess
import itertools
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True)
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--setup", type=int, required=False, default=0)
parser.add_argument("--submit_cmd", type=str, required=False, default="python")
parser.add_argument("--lr",type=float,required=False,default=0.0002) # only used in train_cellpatches
parser.add_argument("--norm",type=str,required=False,default="instance") # only used in train_cellpatches
parser.add_argument("--batch_size",type=int,required=False,default=16) # only used in train_cellpatches
parser.add_argument("--gene",type=str,required=False) # only used in train_cellpatches
parser.add_argument("--continue_train",action='store_true')
parser.add_argument("--epoch_count",default=1,type=int)


def start_training(train_setup,
                   data_roots,
                   netG="resnet_9blocks",
                   in_size=128,
                   continue_train=False,
                   epoch_count=1,
                   submit_cmd="python",
                   input_nc=1,
                   output_nc=1,
                   crop_size=None,
                   preprocess="none",
                   batch_size=1,
                   netD="basic",
                   n_layers_D=3,
                   lr=0.0002,
                   lambda_A=10.0,
                   lambda_B=10.0,
                   lambda_id=0.5,
                   dataset_mode='unaligned',
                   display_scale_min=-1,
                   display_scale_max=1,
                   norm='instance',
                   num_threads=4,
                   n_epochs_decay=100,
                   lr_policy='linear',
                   lr_decay_iter=50,
                   lr_step_gamma=0.1,
                   gene="DTL" # required for cellpatches dataset mode
                   ):

    if crop_size is None:
        crop_size = in_size

    for data_root in data_roots:
        base_cmd = f"{submit_cmd} -u cycle_gan/train.py" +\
                   " --dataroot {} --name {} --input_nc {} --output_nc {} --netG {} --load_size {}"+\
                   " --crop_size {} --checkpoints_dir {} --display_id 0"+\
                   " --preprocess {} --batch_size {} --netD {}"+\
                   " --n_layers_D {} --lr {} --lambda_A {} --lambda_B {}"+\
                   " --lambda_identity {} --dataset_mode {} --display_scale_min {} --display_scale_max {}"+\
                   " --batch_size {} --norm {} --num_threads {}"+\
                   " --lr_policy {} --lr_decay_iter {} --lr_step_gamma {} --n_epochs_decay {}"

        if continue_train:
            base_cmd += f" --continue_train --epoch_count {epoch_count}"

        train_setup_name = "train_s{}".format(train_setup)

        if dataset_mode == 'cellpatches':
            base_cmd += f" --gene {gene}"
            train_setup_name += f"_{gene}"

        checkpoint_dir = os.path.join(os.path.join(data_root, "setups"), 
                                      train_setup_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        cmd = base_cmd.format(data_root,
                              train_setup_name,
                              input_nc,
                              output_nc,
                              netG,
                              in_size,
                              in_size,
                              checkpoint_dir,
                              preprocess,
                              batch_size,
                              netD,
                              n_layers_D,
                              lr,
                              lambda_A,
                              lambda_B,
                              lambda_id,
                              dataset_mode,
                              display_scale_min,
                              display_scale_max,
                              batch_size,
                              norm,
                              num_threads,
                              lr_policy,
                              lr_decay_iter,
                              lr_step_gamma,
                              n_epochs_decay
                              )
        subprocess.Popen(cmd, 
                         shell=True) 

def train_mnist(train_setup, data_root, submit_cmd, args):
    datasets = [f"{i}_{j}" for i,j in list(itertools.combinations([k for k in range(10)], 2))]

    data_roots = []
    for d in datasets:
        data_roots.append(f"{data_root}/{d}")

    start_training(train_setup=train_setup, 
                   data_roots=data_roots,
                   in_size=28, 
                   submit_cmd=submit_cmd)

def train_disc_a(train_setup, data_root, submit_cmd, args):
    datasets = ["0_1"]

    data_roots = []
    for d in datasets:
        data_roots.append(f"{data_root}/{d}")

    start_training(train_setup=train_setup, 
                   data_roots=data_roots,
                   in_size=128,
                   submit_cmd=submit_cmd)


def train_disc_b(train_setup, data_root, submit_cmd, args):
    datasets = ["0_1", "0_2", "1_2"] 

    data_roots = []
    for d in datasets:
        data_roots.append(f"{data_root}/{d}")

    start_training(train_setup=train_setup, 
                   data_roots=data_roots,
                   in_size=128,
                   submit_cmd=submit_cmd)

def train_synapses(train_setup, data_root, submit_cmd, args):
    classes = ["gaba", "acetylcholine", "glutamate","serotonin", "octopamine", "dopamine"]
    datasets = [f"{i}_{j}" for i,j in list(itertools.combinations([k for k in classes], 2))]

    data_roots = []
    for d in datasets:
        data_roots.append(f"{data_root}/{d}")

    start_training(train_setup=train_setup, 
                   data_roots=data_roots,
                   in_size=128,
                   submit_cmd=submit_cmd)

def train_cellpatches(train_setup, data_root, submit_cmd, args):
    start_training(train_setup=train_setup,
                   data_roots=[data_root],
                   in_size=128,
                   submit_cmd=submit_cmd,
                   input_nc=4,
                   output_nc=4,
                   dataset_mode='cellpatches',
                   display_scale_min=0,
                   display_scale_max=0.25,
                   lr=args.lr,
                   batch_size=args.batch_size,
                   norm=args.norm,
                   num_threads=12,
                   n_epochs_decay=5000,
                   lr_policy='step',
                   lr_decay_iter=100,
                   lr_step_gamma=0.5,
                   gene=args.gene,
                   continue_train=args.continue_train,
                   epoch_count=args.epoch_count
                   )

if __name__ == "__main__":
    exp_to_f = {"mnist": train_mnist,
                "synapses": train_synapses,
                "disc_a": train_disc_a,
                "disc_b": train_disc_b,
                "gemelli": train_cellpatches
                }

    args = parser.parse_args()
    f_train = exp_to_f[args.experiment]
    f_train(args.setup, args.data_root, args.submit_cmd, args)
