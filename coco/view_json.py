import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
import os
# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
'''
Coco annotation json checker.
For each dirs, "check" diretory is created containing png imgs with annotation. 
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
# used_for = ["train", "val", "test"]
used_for = ["val"]


for i in used_for:

    # The directory containing the source images
    data_path = f'{coco_dir}/{i}'

    # The path to the COCO labels JSON file
    labels_path = f'{coco_dir}/{i}/annotation_coco.json'

    coco_annotation_file_path = labels_path
    check_path = f'{coco_dir}/{i}/check'

    if not os.path.exists(check_path):
            os.makedirs(check_path)

    coco_annotation = COCO(annotation_file=coco_annotation_file_path)

    indexes = coco_annotation.createIndex()

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

    print(coco_annotation.imgs)
    for key, info in coco_annotation.imgs.items():
        print(f"key: {key} : {info}")
        print(info['file_name'])
        img_id = info['id']
        img_info = coco_annotation.loadImgs([img_id])[0]
        print(img_info)
        img_file_name = img_info["file_name"]
        print(
            f"Image ID: {img_id}, File Name: {img_file_name}"
        )
        ann_ids = coco_annotation.getAnnIds(imgIds=[img_id], iscrowd=None)
        print(f"Ann ids: {ann_ids}")
        # ann_ids.remove(49)
        # ann_ids.remove(50)
        # ann_ids.remove(51)
        anns = coco_annotation.loadAnns(ann_ids)
        # print(f"Anns: {anns}")
        
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

