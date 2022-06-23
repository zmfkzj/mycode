'''
다음의 경우 json에서 삭제
 - category_id가 categories에 없는 annotation들 제거
 - image_id가 images에 없는 annotation들 제거
 - annotations이 없는 categories, images 제거
'''
import json
import chardet

coco_json_path = 'd:/TIPS dataset/export/merged_dataset_coco/annotations/3_clean.json'

with open(coco_json_path, 'r+b') as f:
    encoding = chardet.detect(f.read())['encoding']
with open(coco_json_path, 'r',encoding=encoding) as f:
    coco_json = json.load(f)

# category_id가 categories에 없는 annotation들 제거
cat_list = []
for cat in coco_json['categories']:
    cat_list.append(cat['id'])

new_annotations = []
for obj in coco_json['annotations']:
    if obj['category_id'] in cat_list:
        new_annotations.append(obj)
    else:
        print(f'{obj["category_id"]=}')

coco_json['annotations'] = new_annotations


# image_id가 images에 없는 annotation들 제거
img_list = []
for img in coco_json['images']:
    img_list.append(img['id'])

new_annotations = []
for obj in coco_json['annotations']:
    if obj['image_id'] in img_list:
        new_annotations.append(obj)
    else:
        print(f'{obj["image_id"]=}')

coco_json['annotations'] = new_annotations


# annotations이 없는 categories, images 제거
images_in_anno = set()
cats_in_anno = set()
for obj in coco_json['annotations']:
    images_in_anno.add(obj['image_id'])
    cats_in_anno.add(obj['category_id'])

new_images = []
rm_img_ids = []
for img in coco_json['images']:
    if (img['id'] in images_in_anno) & (img['id'] not in rm_img_ids):
        new_images.append(img)
    else:
        print(f'{img["id"]=}')
coco_json['images'] = new_images

new_cats = []
for cat in coco_json['categories']:
    if cat['id'] in cats_in_anno:
        new_cats.append(cat)
    else:
        print(f'{cat["id"]=}')
coco_json['categories'] = new_cats


with open(coco_json_path,'w') as f:
    json.dump(coco_json,f)
