"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
# import os
from options.test_options import TestOptions
# import json
from data import create_dataset
from models import create_model
# from util.visualizer import save_images
# from util import html_util
import numpy as np
# import ntpath
from tqdm.auto import tqdm
import zarr


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    # opt.num_threads = 0   # test code only supports num_threads = 1
    # opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    z_fake = zarr.open( # zarr store to save fake images and scores
        f'/nrs/funke/funkl/data/dapi/nontargeting_test_samples_to_{opt.gene}.zarr',
        mode='w',
        shape=(len(dataset),4,128,128),
        chunks=(10,4,128,128),
        dtype=np.uint16
    )
    z_fake.attrs['predictions'] = []

    # create a website
    # web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    # if opt.load_iter > 0:  # load_iter is 0 by default
    #     web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    # print('creating web directory', web_dir)
    # webpage = html_util.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in tqdm(enumerate(dataset)):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results dictionary with keys 'real' and 'fake'
        fakes = visuals['fake'].cpu().numpy()
        img_paths = model.get_image_paths()     # get image paths list(?) of returned image paths
        aux_infos = model.get_current_aux_infos()
        aux_infos = dict(aux_infos)
        # aux_infos_json = {v: list(float(k) for k in aux_infos[v][0]) for v in aux_infos.keys()}

        # print(f'output image dtype: {fakes.dtype}, output values dtype: {aux_infos["aux_fake"].dtype}')
        # print(f'output image shape: {visuals["fake"].shape}, output values shape: {aux_infos["aux_fake"].shape}')
        assert len(fakes)==len(aux_infos['aux_real'])==len(aux_infos['aux_fake'])==len(img_paths)
        assert (fakes.dtype==np.float32)
        for fake,aux_real,aux_fake,index in zip(fakes,aux_infos['aux_real'].astype(float),aux_infos['aux_fake'].astype(float),img_paths):
            index = int(index.cpu().numpy())
            fake = np.clip(fake*(2**16-1),0,2**16-1).astype(np.uint16)
            z_fake[index] = fake
            predictions = dict(image_index=index,prediction_real_0=aux_real[0],prediction_real_1=aux_real[1],prediction_fake_0=aux_fake[0],prediction_fake_1=aux_fake[1])
            z_fake.attrs['predictions'] += [predictions]

        
        # short_path = ntpath.basename(img_path[0])
        # name = os.path.splitext(short_path)[0]
        # json.dump(aux_infos_json, open(webpage.get_image_dir() + "/{}_aux.json".format(name), "w+"))

        # if i % 5 == 0:  # save images to an HTML file
        #     print('processing (%04d)-th image... %s' % (i, img_path))
        # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    # webpage.save()  # save the HTML
