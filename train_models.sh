#!/bin/bash

echo "Training Inception V3 multi-class"
python3 train.py -d leaf -m inception_v3 -v

echo "Training Inception V3 binary"
python3 train.py -d leaf -m inception_v3 -v -b

echo "Training ViT multi-class"
python3 train.py -d leaf -m vision_transformer -v

echo "Training ViT binary"
python3 train.py -d leaf -m vision_transformer -v -b
