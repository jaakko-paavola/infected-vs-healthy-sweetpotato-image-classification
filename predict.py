import torch
import click


def predict():


  model = torch.load(model_path)
  model.eval()