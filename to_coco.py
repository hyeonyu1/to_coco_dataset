import os
import cv2
import numpy as np
import pydicom as dcm
import nibabel as nib
from pycocotools import mask as mu
from skimage.transform import resize
from torchvision.transforms import Compose
import torch
import json
from datetime import date

'''
For segmentation.
Only uses dicom file with corresponding nifit file
Need 2 folder in the same dir. "dcm" and "nii"
root
    |- to_coco.py
    |- dcm
    |- nii

Creates dir "coco" with test, train, and val dirs
root
    |- to_coco.py
    |- dcm
    |- nii
    |- coco
        |- test
        |- train
        |- val
        
'''

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class CLAHE():

    def __call__(self, img):
        min_val, max_val = np.percentile(img, [0,100], interpolation="nearest")
        img[img <= min_val] = min_val
        img[img >= max_val] = max_val
        img = (img - min_val)/(max_val - min_val)
        img = (img*255.0).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20,20))
        img_clahe = clahe.apply(img)

        return img_clahe


class Rescale():

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self,img):
        img = (img-img.min())/(img.max())
        w, h = img.shape
        
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size*h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size*w / h
        else:
            new_h,  new_w = self.output_size
        
        new_h, new_w = int(new_h), int(new_w)
        img = resize(img, (new_h, new_w), anti_aliasing=True).astype(np.float32)

        img = np.stack([img, img, img], axis=0)

        return img


class ToTensor():

    def __call__(self, img):
        return torch.from_numpy(img)


def preprocess_image(img):
    preprocessing = Compose([
        CLAHE(),
        Rescale((512, 512)),
        ToTensor()
    ])

    return preprocessing(img.copy()).unsqueeze(0)

#min max noemalization
def dcm_to_png(dcom):
    dcm_data = dcm.read_file(dcom, force=True)

    try:
        if dcm_data.PhotometricInterpretation != "MONOCHROME2":
            image = np.invert(dcm_data.pixel_array.squeeze())
        else:
            image = dcm_data.pxel_array.squeeze()
    except:
        dcm_data.file_meta.TransferSyntaxUID = dcm.uid.ImplicitVRLittleEndian
        if dcm_data.PhotometricInterpretation != "MONOCHROME2":
            image = np.invert(dcm_data.pixel_array.squeeze())
        else:
            image = dcm_data.pixel_array.squeeze()
    image = (image - image.min()) / (image.max() - image.min())

    input_tensor = preprocess_image(image)

    return np.array(input_tensor).squeeze().transpose(1,2,0)


def make_img(dcm_file, coco_dir, used_for):
    img = np.uint8(dcm_to_png(dcm_file)*255)
    cv2.imwrite(f"{coco_dir}/{used_for}/{dcm_file.split('/')[-1].split('.')[0]}.png",img)
    return img.shape[0], img.shape[1], img


def nii_to_seg(mask, w, h):
    gt = nib.load(mask)
    gt_nii = gt.get_fdata()
    gt_nii = np.swapaxes(gt_nii, 0, 1)
    gt_nii = cv2.resize(gt_nii, (w, h))

    seg = []
    y,x = gt_nii.shape[0], gt_nii.shape[1]
    for i in range(0,y):
        for j in range(0,x):
            if gt_nii[i][j] > 0:
                seg.append(j)
                seg.append(i)
    
    return gt_nii, seg


#https://stackoverflow.com/questions/49494337/encode-numpy-array-using-uncompressed-rle-for-coco-dataset
def rle(mask):
    _, binary_mask = cv2.threshold((mask*255).astype('uint8'), 127, 255, cv2.THRESH_BINARY)
    seg = {'counts': [], 'size': list(binary_mask.shape)}
    counts = seg.get('counts')

    last_elem = 0
    running_length = 0

    for i, elem in enumerate(binary_mask.ravel(order='F')):
        if elem == last_elem:
            pass
        else:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1

    counts.append(running_length)

    return seg

def is_crowd(mask):
    path = os.path.dirname(os.path.realpath(__file__))
    coco_dir = path+"/coco"

    _, thresh = cv2.threshold((mask*255).astype('uint8'), 127, 255, cv2.THRESH_BINARY)
    kernal = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(thresh, kernal, iterations=2)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(contours) > 1


def get_bbox(mask):
    pos = np.where(mask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    width = xmax - xmin
    height = ymax - ymin
    return (width*height).item(), [xmin.item(), ymin.item(), width.item(), height.item()], xmax.item(), ymax.item()



def split_dataset():
    path = os.path.dirname(os.path.realpath(__file__))
    dcm_dir = path+"/dcm"
    nii_dir = path+"/nii"
    coco_dir = path+"/coco"

    img = []

    for i in os.listdir(nii_dir):
        filename = i.split(".")[0]
        dcm_file = f'{dcm_dir}/{filename}.dcm'
        if os.path.exists(dcm_file):
            img.append(dcm_file)
    #80 20
    train, test = np.split(np.array(img),[int(len(img)* (.80))])
    #60 20
    train, val = np.split(np.array(img),[int(len(img)* (.75))])

    mean, std = to_coco_notation(train, "train")
    to_coco_notation(val, "val")
    to_coco_notation(test, "test")

    print(f'mean={mean}, std= {std}')



def to_coco_notation(img, used_for):
    coco_dataset = {
        "info": {
                "description": "COCO Style Ankle Fracture Dataset",
                "version": "1.0",
                "year": 2021,
                "contributor": "Hyeon Yu",
                "date_created": date.today().strftime('%Y/%m/%d'),
                "url": "N/A"
        },
        "licenses": [
        {
            "url": "N/A",
            "id": 0,
            "name": "N/A"
        }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {"supercategory": "ankle", "id": 0, "name": "fracture"}
        ]
    }
    
    path = os.path.dirname(os.path.realpath(__file__))
    dcm_dir = path+"/dcm"
    nii_dir = path+"/nii"
    coco_dir = path+"/coco"

    create_dir(f'{coco_dir}/{used_for}')
    
    multi_mask = []
    img_id = 0
    annot_id = 0
    num_train_img = 0

    train_total_img_arr = np.zeros(dcm_to_png(img[0]).shape,np.float64)
    
    for dcm_file in img:
        filename = dcm_file.split('/')[-1].split(".")[0]
        print(f'working on: {filename}\n')
        w, h, img_arr = make_img(dcm_file, coco_dir, used_for)
        
        if used_for == "train":
            train_total_img_arr += img_arr
            num_train_img += 1
        
        image = {
                "license": 0,
                "file_name": f'{filename}.png',
                "id": img_id,
                "height": h,
                "width": w,
                }
        mask, segment = nii_to_seg(f'{nii_dir}/{filename}.nii.gz', w,h)
        area, bbox, xmax, ymax = get_bbox(mask)

        if not is_crowd(mask):
            annotation = {
                            "segmentation": [segment],
                            "area": area,
                            "iscrowd": 0,
                            "image_id": img_id,
                            "bbox": bbox,
                            "category_id": 0,
                            "id": annot_id
                        }

        else:
            area, bbox, xmax, ymax = get_bbox(mask)
            annotation = {
                            "segmentation": rle(mask),
                            "area": area,
                            "iscrowd": 1,
                            "image_id": img_id,
                            "bbox": bbox,
                            "category_id": 0,
                            "id": annot_id
                            }
        coco_dataset["images"].append(image)
        coco_dataset["annotations"].append(annotation)
        img_id+=1
        annot_id+=1

    with open(f'{coco_dir}/{used_for}/annotation_coco.json', 'w') as f:
        json.dump(coco_dataset, f, ensure_ascii=False, indent=4)
    
    return np.mean(train_total_img_arr, axis=(0,1))/num_train_img, np.std(train_total_img_arr, axis=(0,1))/num_train_img



split_dataset()
