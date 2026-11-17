#!/usr/bin/env python3
"""Check the existing runtime without downloading or allocating a model."""
import json
import sys
import numpy
import safetensors
import torch
import transformers
print(json.dumps({'python': sys.version.split()[0], 'torch': torch.__version__,
                  'cuda': torch.version.cuda, 'numpy': numpy.__version__,
                  'transformers': transformers.__version__, 'safetensors': safetensors.__version__,
                  'visible_devices': torch.cuda.device_count(), 'cuda_available': torch.cuda.is_available()}))
