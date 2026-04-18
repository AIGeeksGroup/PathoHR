import pandas as pd
# import sys
# sys.path.append('~/anaconda3/envs/patch/lib/python3.9/site-packages/torch')
import torch
import os
import numpy as np
import pickle
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
from core.utils.learning import set_seed
import wandb
import random
import time
import pdb
from tqdm import tqdm
import json 

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.cuda.amp import autocast, GradScaler

# mutiple GPU initialization
import GPUtil
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from fine_tune_core.core.dataset.dataset import TangleDataset,CustomDataset,featureonly_TangleDataset
from fine_tune_core.core.models.mmssl import MMSSL
from fine_tune_core.core.loss.tangle_loss import InfoNCE, apply_random_mask, init_intra_wsi_loss_function
from fine_tune_core.core.utils.learning import smooth_rank_measure, collate_intra, set_seed
from fine_tune_core.core.utils.process_args import process_args
from fine_tune_core.core.utils.fine_tune_package import write_dict_to_config_file,calculate_metrics,load_json_to_dict

from resnet_custom import resnet50_baseline
from hipt_4k import HIPT_4K
from hipt_model_utils import get_vit256, get_vit4k, eval_transforms



# check the number of available GPU
gpu_count = torch.cuda.device_count()
print(f"Available GPU amounts: {gpu_count}")

for i in range(gpu_count):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Currently used: {torch.cuda.memory_allocated(i) / (1024 ** 2):.2f} MB")
    print(f"  Maximum memory: {torch.cuda.memory_reserved(i) / (1024 ** 2):.2f} MB")


os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BCNB_BREAST_TASKS = ['os']
BREAST_TASKS = {'BCNB': BCNB_BREAST_TASKS}

######
# test
######

def test_loop(clf, ssl_model, test_dataloader):
    
    torch.cuda.empty_cache()
    # set model to eval 
    ssl_model.eval()
    ssl_model.to(DEVICE)

    # do everything without grads 
    with torch.no_grad():
        
        for inputs,val_labels,_,_ in tqdm(test_dataloader):
            # print(len(val_labels)) #64
            val_labels = torch.FloatTensor(val_labels) # float64
            # print(inputs.shape) # [64, 2048, 1024]
            inputs = inputs.unsqueeze(0).permute(1,0,2,3)

            inputs, val_labels = inputs.to(DEVICE), val_labels.cpu()
            inputs = inputs.half()

            wsi_embed = ssl_model(inputs).to(DEVICE)

            print(wsi_embed.shape) 
            # (64,1024)

            wsi_pred_labels = clf.predict(X=wsi_embed.cpu())
            wsi_pred_scores = clf.predict_proba(X=wsi_embed.cpu())
            print(wsi_pred_labels.shape,wsi_pred_scores.shape) 
            # (64,)(64,2)
            cfm, auc, bacc , acc, pre, recall, f1= calculate_metrics(val_labels.cpu().numpy(),wsi_pred_labels, wsi_pred_scores)
            
            wsi_results_dict = {"type":'wsi',"auc": auc, "bacc": bacc, "acc":acc, "pre":pre, "recall":recall, "f1":f1}


    return wsi_results_dict



if __name__ == "__main__":

    args = vars(process_args())
    
    print(args['epochs'])
    
    print(args['method'])

    set_seed(args['seed'])


    # Set params for loss computation 
    RNA_RECONSTRUCTION = True if args["method"] == 'tanglerec' else False 
    INTRA_MODALITY = True if args["method"] == 'intra' else False 
    STOPPING_CRITERIA = 'train_rank' if args["method"] == 'tangle' or args["method"] == 'intra' else 'fixed'
    N_TOKENS_RNA = 4098 if args["study"]=='nsclc' else 4999

    args["rna_reconstruction"] = RNA_RECONSTRUCTION
    args["intra_modality_wsi"] = INTRA_MODALITY
    args["rna_token_dim"] = N_TOKENS_RNA
    
    # Setting path
    # paths 
    ROOT_SAVE_DIR = "./checkpoints/{}_checkpoints_and_embeddings_fine_tune".format(args["study"])
    EXP_CODE = "{}_{}_{}_lr{}_epochs{}_bs{}_tokensize{}_temperature{}_uni".format(
        args['total_feature_path'].split("/")[-1],
        args["method"],
        args["study"],
        args["learning_rate"], 
        args["epochs"], 
        args["batch_size"], 
        args["n_tokens"],
        args["temperature"]
    )
    

    RESULTS_SAVE_PATH = os.path.join(ROOT_SAVE_DIR, EXP_CODE)
    os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)
    write_dict_to_config_file(args, os.path.join(RESULTS_SAVE_PATH, "config.json"))
    
    
    # Set up wandb
    
    wandb.login(key='155515f50d9deacd260e261277070021e3306d66')
    
    wandb.init(project="MedicalHR20250317", 
        # entity="swag162534",
        config = args,
        name = EXP_CODE
        )

    print()
    print(f"Running experiment {EXP_CODE}...")
    print()
    
    # Create a SummaryWriter
    log_dir = os.path.join(ROOT_SAVE_DIR, 'logs', EXP_CODE)
    os.makedirs(log_dir, exist_ok=True)

    
    
    # make tangle dataset
    print("* Setup dataset...")
    # samples =  []
    tangle_dataset = featureonly_TangleDataset(
        feats_dir=args['total_feature_path'], 
        labels_dir = args['total_label_path'],
        sampling_strategy=args["sampling_strategy"], 
        n_tokens=args["n_tokens"]
    )

    
    # dataset = CustomDataset(samples)
    print("Totally matched data:", len(tangle_dataset))

    train_val_dataset, test_dataset,_ = random_split(tangle_dataset,[927,103,6]) # totally matched 1036
    train_dataset,val_dataset= random_split(train_val_dataset,[824,103])


    
    print("* Setup dataloader...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"], 
        shuffle=True, 
        collate_fn=collate_intra,
        num_workers = args['num_workers'],
        pin_memory = args['pin_memory']
    )
    # print(len(train_dataloader))
    
    # set up val dataloader
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args["batch_size"], 
        shuffle=True, 
        collate_fn=collate_intra,
        num_workers = args['num_workers'],
        pin_memory = args['pin_memory']
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args["batch_size"], 
        shuffle=True, 
        collate_fn=collate_intra,
        num_workers = args['num_workers'],
        pin_memory = args['pin_memory']
    )
    
    
    print('initialize LR ...')
    COST = args['embedding_dim']
    CLF = LogisticRegression(C=COST, max_iter=10000, verbose=0, random_state=0)

    
    print('fitting validation model using train data ...')

    input_features = args['embedding_dim'] * args['n_tokens'] 
    output_features = args['embedding_dim']
    linear_layer = nn.Linear(input_features, output_features)


    # for train_patch_embeds,train_labels,_,_ in tqdm(train_dataloader):
    #     train_patch_embeds,train_labels = shuffle(train_patch_embeds,train_labels)
    #     train_patch_embeds = train_patch_embeds.view(train_patch_embeds.size(0),-1)
    #     train_patch_embeds = linear_layer(train_patch_embeds)
        
        # CLF.fit(X=train_patch_embeds.detach().numpy(), y=torch.tensor(train_labels))
        
    print("* Setup model...")
    
    pretrained_weights256 = r'/date2/zhang_h/D/BreastCa/other_models/HIPT-master/HIPT_4K/Checkpoints/vit256_small_dino.pth'
    pretrained_weights4k = r'/date2/zhang_h/D/BreastCa/other_models/HIPT-master/HIPT_4K/Checkpoints/vit4k_xs_dino.pth'
    device256 = torch.device('cuda')
    device4k = torch.device('cuda')

### ViT_256 + ViT_4K loaded independently (used for Attention Heatmaps)
    model256 = get_vit256(pretrained_weights=pretrained_weights256, device=device256)

### ViT_256 + ViT_4K loaded into HIPT_4K API
    ssl_model = HIPT_4K(pretrained_weights256, pretrained_weights4k, device256, device4k).half()
    ssl_model.eval()

    
    wandb.watch(ssl_model)
              
    print('testing  ...')
    
    total = []
    for i in range(10):
        test_result = test_loop(CLF,ssl_model,test_dataloader)
        wandb.log(test_result)
    
    print("All results saved locally.")
    print()
    print("Done.")
    print()
            
