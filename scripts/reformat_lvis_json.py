import json
import os

# Input / output paths
INPUT_JSON = "/home/camposadmin/Documents/lvis/cse_densepose_lvis_v1_ds2_val_v1.json"
OUTPUT_JSON = "/home/camposadmin/Documents/lvis/cse_densepose_lvis_v1_ds2_val_v1_filtered_has_cse.json"


def get_filename_from_id(image_id):
    """
    Default COCO-style filename:
    000000123456.jpg
    """
    return f"{image_id:012d}.jpg"

def main():
    #- 943  # sheep
    #- 1202 # zebra
    #- 569  # horse
    #- 496  # giraffe
    #- 422  # elephant
    #- 80   # cow
    #- 76   # bear
    #- 225  # cat
    #- 378  # dog
    valid_cat_ids = [943, 1202, 569, 496, 422, 80, 76, 225, 378]

    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    
    #Ensure images have a 'file_name' field
    for img in data['images']:
        if "file_name" not in img:
            file_name = get_filename_from_id(img["id"])
            img["file_name"] = file_name
            
    # Ensure annotations have in 'iscrowd' field, set to 0 if absent.       
    for ann in data['annotations']:
        if "iscrowd" in ann:
            continue
        elif "is_crowd" in ann:
            ann["iscrowd"] = ann["is_crowd"]
        else:
            ann["iscrowd"] = 0

    # Filter to keep only valid categories
    filtered_categories = [
        cat for cat in data['categories']
        if cat['id'] in valid_cat_ids
    ]
    
    # Keep annotations of a valid category that have densepose annotations
    filtered_annotations = [
        ann for ann in data['annotations']
        if ann['category_id'] in valid_cat_ids and "dp_vertex" in ann
    ]


    
    # Get image IDs that still have annotations
    keep_image_ids = {ann['image_id'] for ann in filtered_annotations}

    # Filter images
    filtered_images = [
        img for img in data['images']
        if img['id'] in keep_image_ids
    ]

    # Build final dataset
    filtered_data = {
        "info": data['info'],
        "categories": filtered_categories,
        "annotations": filtered_annotations,
        "images": filtered_images,
        "licenses": data['licenses']
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(filtered_data, f)

    print(f"Saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
