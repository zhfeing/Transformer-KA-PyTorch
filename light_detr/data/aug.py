import cv_lib.augmentation.aug as aug


_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
coco_large_train_aug = aug.Compose(
    aug.RandomHorizontalFlip(),
    aug.RandomSelect(
        aug.RandomResize(_scales, max_size=1000),
        aug.Compose(
            aug.RandomResize([400, 500, 600]),
            aug.RandomSizeCrop(384, 600),
            aug.RandomResize(_scales, max_size=1000)
        )
    )
)
coco_large_val_aug = aug.RandomResize([800], max_size=1333)

coco_train_aug = aug.Compose(
    aug.RandomHorizontalFlip(),
    aug.ColorJitter(0.125, 0.5, 0.5, 0.05),
    aug.RandomResize([300, 400, 500]),
    aug.RandomSizeCrop(200, 400),
    aug.RandomResize([300, 480, 512], max_size=512)
)
coco_val_aug = aug.RandomResize([480], max_size=512)

voc_train_aug = aug.Compose(
    aug.RandomHorizontalFlip(),
    aug.ColorJitter(0.125, 0.5, 0.5, 0.05),
    aug.RandomResize([300, 400, 500]),
    aug.RandomSizeCrop(200, 400),
    aug.RandomResize([300, 480, 512], max_size=512)
)
voc_val_aug = aug.RandomResize([300], max_size=512)


__REGISTERED_AUG__ = {
    "coco_train": coco_train_aug,
    "coco_val": coco_val_aug,
    "coco_large_train": coco_large_train_aug,
    "coco_large_val": coco_large_val_aug,
    "voc_train": voc_train_aug,
    "voc_val": voc_val_aug
}


def get_data_aug(dataset_name: str, split: str):
    if "voc" in dataset_name.lower():
        dataset_name = "voc"
    elif "coco_large" in dataset_name.lower():
        dataset_name = "coco_large"
    elif "coco" in dataset_name.lower():
        dataset_name = "coco"
    name = "{}_{}".format(dataset_name, split)
    return __REGISTERED_AUG__[name]
