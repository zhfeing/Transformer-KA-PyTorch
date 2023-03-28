import logging
import copy
from typing import Dict, Any

import torch.utils.data as data

import cv_lib.detection.data as det_data
import cv_lib.distributed.utils as dist_utils
from cv_lib.distributed.sampler import get_train_sampler, get_val_sampler
from cv_lib.classification.data import get_dataset as get_pretrain_dataset

from light_detr.data.aug import get_data_aug
import light_detr.utils as detr_utils
from .patch_dataset import *
from .patch_transforms import get_det_transforms, get_query_transforms


def build_eval_dataset(
    data_cfg: Dict[str, Any],
    val_cfg: Dict[str, Any],
    launch_args: detr_utils.DistLaunchArgs,
):
    logger = logging.getLogger("build_eval_dataset")
    # get dataloader
    data_cfg = copy.deepcopy(data_cfg)
    name = data_cfg.pop("name")
    dataset = det_data.__REGISTERED_DATASETS__[name]
    root = data_cfg.pop("root")
    val_data_cfg = data_cfg.pop("val")
    val_aug = get_data_aug(name, "val")
    data_cfg.pop("train", None)

    val_dataset: det_data.DetectionDataset = dataset(
        root=root,
        augmentations=val_aug,
        **val_data_cfg,
        **data_cfg
    )

    n_classes = val_dataset.n_classes
    if dist_utils.is_main_process():
        logger.info(
            "Loaded %s dataset with %d val examples, %d classes",
            name, len(val_dataset), n_classes
        )
    dist_utils.barrier()
    val_sampler = get_val_sampler(launch_args.distributed, val_dataset)
    val_bs = val_cfg["batch_size"]
    val_workers = val_cfg["num_workers"]
    if launch_args.distributed:
        val_bs, val_workers = dist_utils.cal_split_args(
            val_bs,
            val_workers,
            launch_args.ngpus_per_node
        )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=val_bs,
        num_workers=val_workers,
        pin_memory=True,
        sampler=val_sampler,
        collate_fn=detr_utils.padding_collate_fn
    )
    logger.info(
        "Build validation dataset done\nEval: %d imgs, %d batchs",
        len(val_dataset),
        len(val_loader)
    )
    return val_loader, n_classes


def build_train_dataset(
    data_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    val_cfg: Dict[str, Any],
    launch_args: detr_utils.DistLaunchArgs,
):
    logger = logging.getLogger("build_train_dataset")
    # get dataloader
    train_aug = get_data_aug(data_cfg["name"], "train")
    val_aug = get_data_aug(data_cfg["name"], "val")
    train_dataset, val_dataset, n_classes = det_data.get_dataset(
        data_cfg,
        train_aug,
        val_aug
    )
    if dist_utils.is_main_process():
        logger.info(
            "Loaded %s dataset with %d train examples, %d val examples, %d classes",
            data_cfg["name"], len(train_dataset), len(val_dataset), n_classes
        )
    dist_utils.barrier()
    train_sampler = get_train_sampler(launch_args.distributed, train_dataset)
    val_sampler = get_val_sampler(launch_args.distributed, val_dataset)
    train_bs = train_cfg["batch_size"]
    train_workers = train_cfg["num_workers"]
    val_bs = val_cfg["batch_size"]
    val_workers = val_cfg["num_workers"]
    if launch_args.distributed:
        train_bs, train_workers = dist_utils.cal_split_args(
            train_bs,
            train_workers,
            launch_args.ngpus_per_node
        )
        val_bs, val_workers = dist_utils.cal_split_args(
            val_bs,
            val_workers,
            launch_args.ngpus_per_node
        )
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        num_workers=train_workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=detr_utils.padding_collate_fn,
        drop_last=True
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=val_bs,
        num_workers=val_workers,
        pin_memory=True,
        sampler=val_sampler,
        collate_fn=detr_utils.padding_collate_fn
    )
    logger.info(
        "Build train dataset done\nTraining: %d imgs, %d batchs\nEval: %d imgs, %d batchs",
        len(train_dataset),
        len(train_loader),
        len(val_dataset),
        len(val_loader)
    )
    return train_loader, val_loader, n_classes


def build_pretrain_dataset(
    data_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    logger: logging.Logger,
    launch_args: detr_utils.DistLaunchArgs,
):
    # get dataloader
    unlabeled_dataset = get_pretrain_dataset(data_cfg)
    if data_cfg["name"] == "ImageNet":
        from .filter import delete_instances
        """
        Delete instances with size less than 20 (either width or height)
        to find bad instance, run `scripts/find_bad_instances.py`
        Bad image list:
        train_set/
            n02783161/n02783161_4703.JPEG
            n07760859/n07760859_5275.JPEG
            n04456115/n04456115_4734.JPEG
            n03838899/n03838899_17202.JPEG
            n03838899/n03838899_8045.JPEG
            n03838899/n03838899_2257.JPEG
            n03041632/n03041632_40721.JPEG
            n03041632/n03041632_12338.JPEG
        """
        bad_instances = [1104666, 876220, 876287, 877336, 537427, 641284, 641863, 1224833]
        logger.warning("deleting bad instances: %s", str(bad_instances))
        delete_instances(bad_instances, unlabeled_dataset.samples)
    det_transform = get_det_transforms(unlabeled_dataset.MEAN, unlabeled_dataset.STD)
    query_transform = get_query_transforms(unlabeled_dataset.MEAN, unlabeled_dataset.STD)
    pretrain_dataset = PatchDataset(
        unlabeled_dataset,
        det_transform,
        query_transform
    )
    if dist_utils.is_main_process():
        logger.info(
            "Loaded pretrain %s dataset with %d examples",
            data_cfg["name"], len(pretrain_dataset)
        )
    dist_utils.barrier()
    # get batch sampler
    sampler = get_train_sampler(launch_args.distributed, pretrain_dataset)

    pretrain_bs = train_cfg["batch_size"]
    pretrain_workers = train_cfg["num_workers"]
    if launch_args.distributed:
        pretrain_bs, pretrain_workers = dist_utils.cal_split_args(
            pretrain_bs,
            pretrain_workers,
            launch_args.ngpus_per_node
        )
    pretrain_loader = data.DataLoader(
        pretrain_dataset,
        batch_size=pretrain_bs,
        num_workers=pretrain_workers,
        pin_memory=True,
        sampler=sampler,
        collate_fn=detr_utils.pretrain_padding_collate_fn,
        drop_last=True
    )
    logger.info(
        "Build dataset done\nTraining: %d imgs, %d batchs",
        len(pretrain_dataset),
        len(pretrain_loader),
    )
    return pretrain_loader
