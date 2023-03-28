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
    --file-name-cfg Amalgamate \
    --cfg-filepath config/voc/amalgamation/resnet50-amg-seq-task-no_cross.yaml \
    --log-dir run/voc/amalgamation/resnet50-amg-seq-task-no_cross \
    --worker amalgamate &
