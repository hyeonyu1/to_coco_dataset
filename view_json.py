import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import requests
from pycocotools.coco import COCO
import os

'''
Coco annotation json checker, goes into test, train, val dirs and creates check dir 
containing png image and another img with bounding box and segmentation.
Code based on https://leimao.github.io/blog/Inspecting-COCO-Dataset-Using-COCO-API/
Root
    |- coco
        |- test
        |- train
        |- val
        |- view_json.py
to 
Root
    |- coco
        |- test
            |- check
        |- train
            |- check
        |- val
            |- check
        |- view_json.py
'''


coco_dir = os.path.dirname(os.path.realpath(__file__))
used_for = ["train", "val", "test"]

for i in used_for:

    # The directory containing the source images
    data_path = f'{coco_dir}/{i}'

    # The path to the COCO labels JSON file
    labels_path = f'{coco_dir}/{i}/annotation_coco.json'

    instances_json_path = labels_path
    # images_path = data_path
    annotation_path = labels_path
    # image_dir = data_path
    coco_annotation_file_path = labels_path
    check_path = f'{coco_dir}/{i}/check'

    if not os.path.exists(check_path):
            os.makedirs(check_path)

    coco_annotation = COCO(annotation_file=coco_annotation_file_path)


    # Category IDs.
    cat_ids = coco_annotation.getCatIds()
    print(f"Number of Unique Categories: {len(cat_ids)}")
    print("Category IDs:")
    print(cat_ids)  # The IDs are not necessarily consecutive.

    # All categories.
    cats = coco_annotation.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]
    print("Categories Names:")
    print(cat_names)

    # Category ID -> Category Name.
    query_id = cat_ids[0]
    query_annotation = coco_annotation.loadCats([query_id])[0]
    query_name = query_annotation["name"]
    query_supercategory = query_annotation["supercategory"]
    print("Category ID -> Category Name:")
    print(
        f"Category ID: {query_id}, Category Name: {query_name}, Supercategory: {query_supercategory}"
    )

    # Category Name -> Category ID.
    query_name = cat_names[0]
    query_id = coco_annotation.getCatIds(catNms=[query_name])[0]
    print("Category Name -> ID:")
    print(f"Category Name: {query_name}, Category ID: {query_id}")

    # Get the ID of all the images containing the object of the category.
    img_ids = coco_annotation.getImgIds(catIds=[query_id])
    print(f"Number of Images Containing {query_name}: {len(img_ids)}")

    for i in range(0,len(img_ids)):
        print("\n\n")
        # Pick one image.
        img_id = img_ids[i]
        img_info = coco_annotation.loadImgs([img_id])[0]
        print(img_info)
        img_file_name = img_info["file_name"]
        print(
            f"Image ID: {img_id}, File Name: {img_file_name}"
        )

        # Get all the annotations for the specified image.
        ann_ids = coco_annotation.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco_annotation.loadAnns(ann_ids)

        print(f"Annotations for Image ID {img_id}:")

        # # Use URL to load image.
        im = Image.open(f'{data_path}/{img_file_name}')
        # Save image and its labeled version.
        plt.axis("off")
        plt.imshow(np.asarray(im))

        file_name = img_file_name.split(".")[0]
        plt.savefig(f"{check_path}/{file_name}.png", bbox_inches="tight", pad_inches=0)
        # Plot segmentation and bounding box.
        coco_annotation.showAnns(anns, draw_bbox=True)
        plt.savefig(f"{check_path}/{file_name}_annotated.png", bbox_inches="tight", pad_inches=0)
        plt.clf()
