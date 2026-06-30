SAM3 installation / setup [here](./README_SAM3.md)

## Running DensePose Training

The config [`sam3_densepose_simple.yaml`](./sam3/train/configs/lvis/sam3_densepose_simple.yaml) is the main config for training the SAM3 DensePose head. To run on your own data, you need to modify several sections of this file. The key areas are outlined below.

### 1. Prepare Your Data

Your custom dataset must be in **COCO-style JSON format** with DensePose CSE annotations. Each annotation should include:
- Standard COCO fields (`bbox`, `segmentation`, `category_id`, `image_id`)
- CSE embedding fields used by the DensePose head (e.g., `dp_vertex`, `dp_x`, `dp_y`, `ref_model`)

You will need separate JSON files for training and validation splits, plus the corresponding image directories.

### 2. Update Data Paths

Update the `img_folder` and `ann_file` entries under both the **train** and **val** data sections to point to your custom data:

```yaml
# ---- Training data (trainer.data.train.dataset) ----
img_folder: /path/to/your/train/images
ann_file: /path/to/your/train_annotations.json

# ---- Validation data (trainer.data.val.dataset) ----
img_folder: /path/to/your/val/images
ann_file: /path/to/your/val_annotations.json
```

Also update the ground-truth path used by the **evaluator** under `trainer.meters.val.lvis.detection.pred_file_evaluators`:

```yaml
gt_path: /path/to/your/val_annotations.json
```

### 3. Update Model Checkpoint Paths

The model section (`trainer.model`) references two pre-trained checkpoint files. Update these to match your local paths:

```yaml
# Main SAM3 model checkpoint (pre-trained weights to initialize from)
model_init_path: /path/to/your/sam3_checkpoint.pt

# DensePose head initialization weights (Detectron2 DensePose model)
dp_init_path: /path/to/your/model_final_densepose.pkl
```
If neither of these is specified, by default SAM3 pretrained weights will be used with no densepose pretraining.

### 4. Update CSE Mesh Vertex Feature Paths

The `cse_embedder` section under `trainer.model` defines mesh specifications for each animal/object category. Each mesh entry has an `INIT_FILE` that points to a `.pkl` file containing the pre-computed vertex features for that mesh. Update every `INIT_FILE` path:

```yaml
cse_embedder:
  mesh_specs:
    "cat_7466":
      INIT_FILE: "/path/to/your/mesh_vert_feats/phi_cat_7466_256.pkl"
    "dog_7466":
      INIT_FILE: "/path/to/your/mesh_vert_feats/phi_dog_7466_256.pkl"
    # ... update all other mesh entries similarly
```

If you are training on a **custom set of categories**, you should:
- Remove mesh entries for categories not in your data
- Add new mesh entries for your custom categories (you will need to generate the corresponding vertex feature `.pkl` files)
- Update `NUM_VERTICES` and `FEATURE_DIM` to match your meshes

Additionally, modify the [`builtin.py`](./sam3/train/data/builtin.py) file to include your mesh and associated geodists file.

### 5. Update the Category-to-Mesh Mapping for Evaluation

The `per_point_gps` meter under `trainer.meters.val.lvis` maps COCO/LVIS category IDs to mesh names. Update `cat_to_mesh` to reflect the categories in your dataset:

```yaml
per_point_gps:
  cat_to_mesh:
    <your_cat_id_1>: "your_mesh_name_1"
    <your_cat_id_2>: "your_mesh_name_2"
    # ...
```

### 6. Adjust Training Parameters and Losses (Optional)

To configure which losses apply and with what weights, edit the lvis_train.loss section of the config. Losses can also be added here. Depending on compute limitations, modify the scratch.train_batch_size field. Keep gradient_accumulation_steps equal to num_chunks. The effective minibatch size is train_batch_size divided by num_chunks.  


### 7. Configure Unfrozen Parameters

The `unfrozen_prefixes` list under `trainer.model` controls which parts of the model are trained (everything else is frozen). The defaults are:

```yaml
unfrozen_prefixes:
  - "densepose_head"
  - "cse_embedder"
  - "segmentation_head"
  - "dot_prod_scoring"
  - "transformer.decoder.bbox_embed"
```

If you want to fine-tune additional components (e.g., the vision backbone), add the corresponding prefix to this list.

### 8. Update Output and Logging Paths

```yaml
# Top-level output directory
paths:
  experiment_log_dir: "/path/to/your/experiment/logs"

# BPE vocabulary (should already exist under sam3/assets/)
paths:
  bpe_path: "./sam3/assets/bpe_simple_vocab_16e6.txt.gz"
```

By default, the first 5 batches with GT data are plotted and saved in the experiment logging directories. Modify the following if you want to change the default location/frequency.

```yaml
trainer:
  logging:
    vis_dir: path/to/desired/folder
    num_vis_batches:
```

### 9. Run Training

```bash
python sam3/train/train.py -c configs/lvis/sam3_densepose_simple.yaml --use-cluster 0 --num-gpus 1
```

Adjust `--num-gpus` based on your hardware. To run on a SLURM cluster, set `--use-cluster 1` and configure the `submitit` section in the YAML (partition, qos, etc.). (I haven't tested this but this is from the SAM3 setup)

By default, validation runs every epoch, modify val_epoch_freq as needed. Validation epochs will create a GPS/GPSM_matched_predictions.pkl file in the log_dir. This can be used to visualize predictions with the visualize_cse_predictions.py script.

```
python scripts/visualize_cse_predictions.py --predictions log_dir/GPSM_matched_predictions.pkl --images_dir /path_to_val_images --lbo_dir /path_to_lbo_feat_folder --output_dir /desired_path_to_output_visualizations --alpha .7
``` 


