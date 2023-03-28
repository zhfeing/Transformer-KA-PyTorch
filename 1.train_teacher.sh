#!/bin/bash


CV_LIB_PATH=/c22940/zhf/zhfeing_cygpu1/project/cv-lib-PyTorch
export PYTHONPATH=$CV_LIB_PATH:./


export CUDA_VISIBLE_DEVICES="0,1"
port=9003

python dist_engine.py \
    --num-nodes 1 \
    --rank 0 \
    --master-url tcp://localhost:$port \
    --backend nccl \
    --multiprocessing \
    --file-name-cfg Teacher-1 \
    --cfg-filepath config/voc/multitask/resnet50-t1.yaml \
    --log-dir run/voc/multitask/resnet50-t1 \
    --worker train &

export CUDA_VISIBLE_DEVICES="2,3"
port=9005
python dist_engine.py \
    --num-nodes 1 \
    --rank 0 \
    --master-url tcp://localhost:$port \
    --backend nccl \
    --multiprocessing \
    --file-name-cfg Teacher-2 \
    --cfg-filepath config/voc/multitask/resnet50-t2.yaml \
    --log-dir run/voc/multitask/resnet50-t2 \
    --worker train &
