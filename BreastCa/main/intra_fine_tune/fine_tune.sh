#!/bin/bash
# python3 fine_tune_os.py --method "tangle" --gpu_devices 0
# python3 fine_tune_os.py --method "tanglerec" --gpu_devices 0
# python intra_fine_tune.py --method "intra" --gpu_devices 0 --batch_size 64 --epochs 1000
# python intra_fine_tune_resnet50.py --method "intra" --gpu_devices 0 --batch_size 64 --epochs 1000

python intra_fine_tune.py --method "intra" --gpu_devices 0 --batch_size 64 --epochs 1000