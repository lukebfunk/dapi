### GOYAL 

1. install dependencies 

`pip install -r requirements.txt`


2. runing analysis 
to run permutation analysis use `counterfactual_visual_explanation.py`

```bash
--model type:       VGG or RES type of model to use
--dataset name:     [synapses, mnist, apples_oranges, disc_a, disc_b, 
                    summer_winter, horses_zebras]
--class_names       class name to use for feature swap 

--folder_name:      folder name where data is located
--query_image_list: path to txt file that contains path to images that
                    were used for GAN  translation

--config_name:      configuration file for each dataset
--chk_name:         path to checkpoint
--distractor_folder: path to raw images folder
--flatten:          for 1 chanell image set it to False
--solvers:          [exhaustive, continues] 
```


### Example on APPLE and ORANGES
to recreate apple and oranges results:
```bash
python counterfactual_visual_explanation.py  --model_type VGG --model_name Vgg2D  --dataset_name apples_oranges  --class_names apples_oranges  --query_image_list apples_oranges/reals.txt  --folder_name data  --config_name configs/apples_oranges.ini  --chk_name apples_oranges/classifier/vgg_checkpoint  --distractor_folder apples_oranges/train/ --flatten False --solvers continues

python counterfactual_visual_explanation.py  --model_type VGG --model_name Vgg2D  --dataset_name apples_oranges  --class_names oranges_apples  --query_image_list apples_oranges/reals.txt  --folder_name  data  --config_name configs/apples_oranges.ini  --chk_name apples_oranges/classifier/vgg_checkpoint  --distractor_folder apples_oranges/train/ --flatten False --solvers continues

python counterfactual_visual_explanation.py  --model_type RES --model_name ResNet  --dataset_name apples_oranges  --class_names apples_oranges  --query_image_list apples_oranges/reals.txt  --folder_name data  --config_name configs/apples_oranges.ini  --chk_name apples_oranges/classifier/res_checkpoint  --distractor_folder apples_oranges/train/ --flatten False --solvers continues

python counterfactual_visual_explanation.py  --model_type RES --model_name ResNet  --dataset_name apples_oranges  --class_names oranges_apples  --query_image_list apples_oranges/reals.txt  --folder_name  data  --config_name configs/apples_oranges.ini  --chk_name apples_oranges/classifier/res_checkpoint  --distractor_folder apples_oranges/train/ --flatten False --solvers continues
```

after running the script it will create a local directory that will contain a `prediction.csv` and `.png` images corresponding to iterative swap of a feature according to Goyal `Algorithm 1`
