SAM3 installation / setup [here](./README_SAM3.md)

## Running SAM3 on LVIS animal data
First download coco2017 images from [COCO](https://cocodataset.org/#download). 
The config file to evaluate LVIS data with SAM3 requires all images from train/val splits to be in the same directory.
Additionally, gather DensePose LVIS ds2 train/val annotations from [here](https://github.com/facebookresearch/detectron2/blob/8a9d885b3d4dcf1bef015f0593b872ed8d32b4ab/projects/DensePose/doc/DENSEPOSE_DATASETS.md#continuous-surface-embeddings-annotations-3).
  
By default the LVIS json files need some reformatting to be used with SAM3 (some fields need to be populated, and we trim down to only the animal categories).
A filtered json for eval can be found [here](./sam3/train/configs/lvis/cse_densepose_lvis_v1_ds2_val_v1_filtered.json).
This was created by running   
```
python scripts/reformat_lvis_json.py
```
If you need a different category split you can edit the ```valid_cat_ids``` variable. 

Before evaluating on LVIS data, make sure all file paths in the yaml files [here](./sam3/train/configs/lvis) are appropriate for your setup, which would require updating the all ```img_path``` vars to point to your combined COCO image folder. All ```ann_file``` and ```gt_path``` vars should also be updated to the location of your desired json annotations file.

Finally, to evaluate choose either the ```bbox``` or ```segm``` yaml configs and run following the SAM3 readme instructions.

For example:
```
python sam3/train/train.py -c configs/lvis/default_lvis_config_eval_bbox.yaml --use-cluster 0 --num-gpus 1
```
