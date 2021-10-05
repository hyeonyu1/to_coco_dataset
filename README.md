# to_coco_dataset
Converts dicom file with nifit ground truth segmentation to coco dataset format
to_coco.py looks for dcm and nii directories in the same folder.
view_json.py can be run after running to_coco.py and is inside the coco dir.
view_json.py returns the normal image and image with bounding box and segmentation from the coco annotation.
