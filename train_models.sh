#!/bin/bash

echo "Training plant models"

echo "Training ResNet18 multiclass for plant"

python3 train.py -m resnet18 -d plant

echo "Training ResNet18 binary for plant"

python3 train.py -m resnet18 -d plant -b -bl 3

echo "Training Inception V3 multiclass for plant"

python3 train.py -m inception_v3 -d plant

echo "Training Inception V3 binary for plant"

python3 train.py -m inception_v3 -d plant -b -bl 3

echo "Training Vision Transformer multiclass for plant"

python3 train.py -m vision_transformer -d plant

echo "Training Vision Transformer binary for plant"

python3 train.py -m vision_transformer -d plant -b -bl 3

echo "Training leaf models"

echo "Training ResNet18 multiclass for leaf"

python3 train.py -m resnet18 -d leaf

echo "Training ResNet18 binary for leaf"

python3 train.py -m resnet18 -d leaf -b -bl 3

echo "Training Inception V3 multiclass for leaf"

python3 train.py -m inception_v3 -d leaf 

echo "Training Inception V3 binary for leaf"

python3 train.py -m inception_v3 -d leaf -b -bl 3

echo "Training Vision Transformer multiclass for leaf"

python3 train.py -m vision_transformer -d leaf

echo "Training Vision Transformer binary for leaf"

python3 train.py -m vision_transformer -d leaf -b -bl 3

