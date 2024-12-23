#!/usr/bin/env python

import sys
import os
import torch
import gymnasium as gym

CHECK_POINTS_PATH = os.path.join("..", "checkpoints")

x = torch.rand(1, 5)
print(x)

