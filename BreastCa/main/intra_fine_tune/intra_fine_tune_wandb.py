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

    
####################
# fine tune training
####################


def train_loop(args, loss_fn_interMod, loss_fn_rnaRecon, loss_fn_intraMod, ssl_model, epoch, dataloader, optimizer, scheduler_warmup, scheduler):
    
    torch.cuda.empty_cache()
    
    ssl_model.train()
    ssl_model.to(DEVICE)

    ep_loss, ep_recon_loss, ep_inter_loss, ep_intra_loss = 0., 0., 0., 0.
    fb_time = 0.
    wsi_embeds = []
    
    # scaler = GradScaler('cuda')
    
    for b_idx, (patch_emb, train_labels,patch_emb_aug, avg_patch_emb) in enumerate(dataloader):
        
        # print(patch_emb,rna_seq,patch_emb_aug,avg_patch_emb)
        
        losses = []    
        s_fb = time.time()
        
        # print(patch_emb.shape,rna_seq.shape,patch_emb_aug.shape,avg_patch_emb.shape)
        
        # rna_seq =rna_seq.repeat(1,N_TOKENS_RNA)
        # print(rna_seq.shape)
        rna_seq = None
        # preprocessing for intra-modality loss 
        if args["intra_modality_wsi"]:
            if args["intra_modality_mode_wsi"] == "contrast_token_views":
                patch_emb = torch.cat((patch_emb, patch_emb_aug))
            elif args["intra_modality_mode_wsi"] == "reconstruct_masked_emb" or args["intra_modality_mode_wsi"] == "reconstruct_masked_emb+contrast_avg_emb":
                patch_emb_mask = apply_random_mask(patch_embeddings=patch_emb, percentage=args['mask_percentage'])
                patch_emb = torch.cat((patch_emb, patch_emb_mask))
                
        # # set the patch embedding for validation        
        # args['patch_emb'] = patch_emb

        # set data on device 
        patch_emb = patch_emb.to(DEVICE)
        rna_seq = rna_seq.to(DEVICE) if rna_seq is not None else rna_seq
        # print(patch_emb.shape,rna_seq.shape)
        if args["intra_modality_mode_wsi"] == "contrast_avg_emb" or args["intra_modality_mode_wsi"] == "reconstruct_avg_emb" or args["intra_modality_mode_wsi"] == "reconstruct_masked_emb+contrast_avg_emb":
            avg_patch_emb = avg_patch_emb.cuda()
            
        print('forwarding ... ')
        # forward pass and loss 
        if args["intra_modality_wsi"]:
            wsi_emb, rna_emb, rna_reconstruction = ssl_model(patch_emb, None)
        else:
            wsi_emb, rna_emb, rna_reconstruction = ssl_model(patch_emb, rna_seq)
            
        print('moving on model ... ')
        # intra modality loss wsi <-> wsi
        if rna_emb is None and rna_reconstruction is None:
            if args["intra_modality_mode_wsi"] == "contrast_token_views":
                split_idx = int(patch_emb.shape[0]/2)
                losses.append(loss_fn_intraMod(query=wsi_emb[:split_idx], positive_key=wsi_emb[split_idx:], symmetric=args["symmetric_cl"])) # 1. first set of token views 2. second set of token views (augmentation)
            elif args["intra_modality_mode_wsi"] == "contrast_avg_emb":
                losses.append(loss_fn_intraMod(query=wsi_emb, positive_key=avg_patch_emb, symmetric=args["symmetric_cl"]))
            elif args["intra_modality_mode_wsi"] == "reconstruct_avg_emb":
                losses.append(loss_fn_intraMod(wsi_emb, avg_patch_emb))
            elif args["intra_modality_mode_wsi"] == "reconstruct_masked_emb":
                split_idx = int(patch_emb.shape[0]/2)
                losses.append(loss_fn_intraMod(wsi_emb[split_idx:], wsi_emb[:split_idx])) # 1. masked wsi_emb 2. umasked wsi_emb
            elif args["intra_modality_mode_wsi"] == "reconstruct_masked_emb+contrast_avg_emb":
                split_idx = int(patch_emb.shape[0]/2)
                losses.append(loss_fn_intraMod(wsi_emb[split_idx:], wsi_emb[:split_idx])) # 1. masked wsi_emb 2. umasked wsi_emb
                losses.append(loss_fn_intraMod(query=wsi_emb[:split_idx], positive_key=avg_patch_emb, symmetric=args["symmetric_cl"]))
            else:
                raise ValueError("Invalid intra_modality_mode_wsi.")
            ep_intra_loss += losses[-1].item()
            
        print('adding loss ...')
        
        # inter modality loss wsi <-> rna
        if rna_emb is not None:
            
            # print(wsi_emb.shape,rna_emb.shape)
            wsi_emb = wsi_emb.unsqueeze(0).repeat(N_TOKENS_RNA,1)
            # print(wsi_emb.shape,rna_emb.shape)

            losses.append(loss_fn_interMod(query=wsi_emb, positive_key=rna_emb, symmetric=args["symmetric_cl"]))
            ep_inter_loss += losses[-1].item()
            
 
        # intra modality loss rna <-> rna
        if rna_reconstruction is not None:
            print('reconstruction ... ')  
            losses.append(loss_fn_rnaRecon(rna_reconstruction, rna_seq))
            ep_recon_loss += losses[-1].item()
        
        print('backward ... ')
        loss = sum(losses)
        optimizer.zero_grad()

        # scaler.scale(loss).backward()
        loss.backward()
        
        # allow small grads accumulation
        # if (b_idx + 1) % 4 == 0:
            
        # scaler.step(optimizer)
        # scaler.update()
        optimizer.step()
        optimizer.zero_grad()
        
        # Log loss to wandb
        wandb.log({"train epoch": epoch, "loss": loss.item()})

        e_fb = time.time()
        fb_time += e_fb - s_fb

        if epoch <= args["warmup_epochs"]:
            scheduler_warmup.step()
        else:
            scheduler.step()  
            
        if (b_idx % 3) == 0:
            print(f"Loss for batch: {b_idx} = {loss}")
            
        ep_loss += loss.item()
        
        # get the train embeds to calculate rank
        ssl_model.eval()
        # do everything without grads 
        with torch.no_grad():
            wsi_emb_to_store, rna_emb_to_store, _ = ssl_model(patch_emb)
            wsi_embeds.extend(wsi_emb_to_store.detach().cpu().numpy())
            # rna_embeds.extend(rna_emb_to_store.detach().cpu().numpy())
            # print(type(all_embeds),len(all_embeds))
        ssl_model.train()
    
    # track rank
    all_embeds_tensor = torch.Tensor(np.array(wsi_embeds)).unsqueeze(1)
    # rna_embeds_tensor = torch.Tensor(np.array(rna_embeds)).unsqueeze(0)
    # print(wsi_embeds_tensor.shape,rna_embeds_tensor.shape)
    # all_embeds_tensor = torch.matmul(wsi_embeds_tensor,rna_embeds_tensor)
    # print(all_embeds_tensor.shape)
    rank = smooth_rank_measure(all_embeds_tensor)  
        
    return ep_loss, rank, ssl_model

#############
# evaluation
#############

def val_loop(clf, ssl_model, val_dataloader):
    
    # set model to eval 
    ssl_model.eval()
    ssl_model.to(DEVICE)
    
    # do everything without grads 
    with torch.no_grad():
        
        for inputs,val_labels,_,_ in tqdm(val_dataloader):
            # print(len(val_labels)) #64
            val_labels = torch.IntTensor(val_labels) # float64
            # print(inputs.shape,rna.shape)
            inputs, val_labels = inputs.to(DEVICE), val_labels.cpu()
            wsi_embed, rna_embed, _ = ssl_model(inputs)
            # print(wsi_embed.shape) # (64,1024)

            wsi_pred_labels = clf.predict(X=wsi_embed.cpu())
            wsi_pred_scores = clf.predict_proba(X=wsi_embed.cpu())
            # print(wsi_pred_labels.shape,wsi_pred_scores.shape) # (64,)(64,2)
            cfm, auc, bacc , acc, pre, recall, f1= calculate_metrics(val_labels.cpu().numpy(),wsi_pred_labels, wsi_pred_scores)
            # print(cfm)
            # print(wsi_results)
            wsi_results_dict = {"type":'wsi',"auc": auc, "bacc": bacc, "acc":acc, "pre":pre, "recall":recall, "f1":f1}

    return wsi_results_dict

######
# test
######

def test_loop(clf, ssl_model, test_dataloader):
    
    # set model to eval 
    ssl_model.eval()
    ssl_model.to(DEVICE)
    count = 0

    # do everything without grads 
    with torch.no_grad():
        
        for inputs,val_labels,_,_ in tqdm(test_dataloader):
            # print(len(val_labels)) # 64
            val_labels = torch.IntTensor(val_labels) # float64
            # print(inputs.shape) # [64, 2048, 1024]
            # inputs = inputs.unsqueeze(0).permute(1,0,2,3)
            #print(inputs.shape)
            inputs, val_labels = inputs.to(DEVICE), val_labels.cpu()
            
            wsi_embed, rna_embed, _ = ssl_model(inputs)
            # wsi_embed = ssl_model(inputs)
            
            #print(wsi_embed.shape) 
            # (64,1024) tangle & (1,1024) resnet50

            wsi_pred_labels = clf.predict(X=wsi_embed.cpu())
            wsi_pred_scores = clf.predict_proba(X=wsi_embed.cpu())
            #print(wsi_pred_labels.shape,wsi_pred_scores.shape) 
            count= count +1
            print("count = ",count)
            # (64,)(64,2) tangle
            # (1,)(1,2) resnet50
            cfm, auc, bacc , acc, pre, recall, f1= calculate_metrics(val_labels.cpu().numpy(),wsi_pred_labels, wsi_pred_scores)
            
            wsi_results_dict = {"type":'wsi',"auc": auc, "bacc": bacc, "acc":acc, "pre":pre, "recall":recall, "f1":f1}


    return wsi_results_dict


# def collate_all_feat(batchs:list):# list of lists
    
#     feats = []
#     # print(len(batchs))
#     print('generating feature embedding for LG ...')
#     for batch in tqdm(batchs):
        
#     #!!! make sure the following shape are fixed:
#     #!!! dim: 2 , 1 , 2 , 1
    
#         target_shapes = [(1024, 1024),(4999,), (1024, 1024), (1024,) ]
    
#         shape_to_tensor = {tuple(t.shape): t for t in batch}

#         batch = [shape_to_tensor[shape] for shape in target_shapes]

#         #feat = [torch.tensor(x).view(-1) for x in zip(*batch[0])]
#         feat = torch.tensor(batch[0]).view(-1)
#         feats.append(feat)
        
#     samples = torch.stack(feats) # (740,1024)
#     print(samples.shape)
     
#     return samples

# def obtain_feat(dataset_):
#     batchs = []
#     # print(dataset.__len__(True))
#     print('obtaining features ...')
#     for i in tqdm(range(len(dataset_))):
#         # print("No.",i)
#         batch = list(dataset_[i])
#         # print(len(batch))
#         batchs.append(batch)
#     print('done')
#     # print(len(batchs))
        
#     patch_embeds = collate_all_feat(batchs) 
        
#     print(patch_embeds)
    
#     return patch_embeds



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
    
    # for i in range(tangle_dataset.__len__(True)):
    #     # print('loading patch {}'.format(i))
    #     samples.append(tangle_dataset[i]) # tuple
    
    # total_labels = pd.read_csv(args['total_label_path'])
    # total_labels = torch.Tensor(np.array(total_labels['os'].values))
    
    # dataset = CustomDataset(samples)
    print("Totally matched data:", len(tangle_dataset))
    
    # train_val_dataset, test_dataset,_ = random_split(tangle_dataset,[936,104,6]) # totally matched 1046
    # train_dataset,val_dataset= random_split(train_val_dataset,[832,104])

    train_val_dataset, test_dataset,_ = random_split(tangle_dataset,[927,103,6]) # totally matched 1036
    train_dataset,val_dataset= random_split(train_val_dataset,[824,103])


    # datasampler = DistributedSampler(train_dataset, num_replicas=GPU_NUM, rank=RANK)
    # datasampler = DistributedSampler(val_dataset, num_replicas=GPU_NUM, rank=RANK)
    # datasampler = DistributedSampler(test_dataset, num_replicas=GPU_NUM, rank=RANK)

    
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
    
    # print('* Setup labels ... ')
    # preparing labels for each part
    
    # # loading train labels for fit
    # train_labels = pd.read_csv(args['train_label_path'])
    # train_labels = torch.Tensor(np.array(train_labels['os_x'].values)) # (740,)
    # # loading val and test labels
    # val_labels = pd.read_csv(args['val_label_path'])
    # val_labels = torch.Tensor(np.array(val_labels['os_x'].values)) # (92,)
    # test_labels = pd.read_csv(args['test_label_path'])
    # test_labels = torch.Tensor(np.array(test_labels['os_x'].values)) # (94,)


    # print('extract patches from datasets for fittng and prediction ...')
    
    # print('loading training dataset for LR ... ')
    # train_patch_embeds = obtain_feat(train_dataset)
    
    
    # with open(args['train_patch_embeds'], 'wb') as f:
    #     pickle.dump(train_patch_embeds, f)
        
    # print('all feature saved.')
    
    
    # print('loading features ... ')
    
    # with open(args['train_patch_embeds'], 'rb') as f:
    #     train_patch_embeds = pickle.load(f)
    # print(train_patch_embeds.shape)
    
    
    print('initialize LR ...')
    # NUM_C = 2
    # COST = (train_patch_embeds.shape[1] * NUM_C) / 100
    # print(COST)
    COST = args['embedding_dim']
    CLF = LogisticRegression(C=COST, max_iter=10000, verbose=0, random_state=0)
    # print(patch_embeds[0].shape) # 1024
    
    print('fitting validation model using train data ...')

    input_features = args['embedding_dim'] * args['n_tokens'] 
    output_features = args['embedding_dim']
    linear_layer = nn.Linear(input_features, output_features)


    for train_patch_embeds,train_labels,_,_ in tqdm(train_dataloader):
        # print(train_labels)
        # print(train_patch_embeds.shape) #(64,1024,1024)
        # print(len(train_labels)) # 64        
        # print(train_patch_embeds.shape)
        train_patch_embeds,train_labels = shuffle(train_patch_embeds,train_labels)
        # print(train_patch_embeds.shape)
        
        # apply linear switch
        train_patch_embeds = train_patch_embeds.view(train_patch_embeds.size(0),-1)
        # print(train_patch_embeds.shape) #(64,1048576) # n_token=1024
        # print(train_patch_embeds.shape) # [256, 2097152] # n_token=2048
        train_patch_embeds = linear_layer(train_patch_embeds)
        # print(train_patch_embeds.shape) # (64,512)  # 1024

        # print(torch.tensor(train_labels).shape) #(64,)
        # print(np.unique(np.array(train_labels)))
        
        
        CLF.fit(X=train_patch_embeds.detach().numpy(), y=torch.tensor(train_labels))

    
    # set up model config, n_tokens_wsi, n_tokens_rna, patch_embedding_dim=768
    # print("* Setup model...")
    # ssl_model = resnet50_baseline(pretrained=False)
    ssl_model = MMSSL(config=args, n_tokens_rna=N_TOKENS_RNA)

    print("* Checking for the checkpoint ... ")

    MODELS = {'intra': args['pretrained']}
    print("RESULTS_SAVE_PATH = ",RESULTS_SAVE_PATH+"model.pt")
    try:
        state_dict = torch.load(os.path.join(RESULTS_SAVE_PATH, "model.pt"))
        print("RESULTS_SAVE_PATH = ",RESULTS_SAVE_PATH+"model.pt")
        # tangle original
        # state_dict = torch.load(args['pretrained'])
        ssl_model.load_state_dict(state_dict, strict=False)
        log = load_json_to_dict(os.path.join(RESULTS_SAVE_PATH, "log.json"))
        ("* Found checkpoint. Loading Now ... ")
        begin_epoch = log['epoch']
        print(" Training will begin with epoch {}".format(begin_epoch))

    except FileNotFoundError:
        print("* No checkpoint found, begin new training ...")
        begin_epoch = 0
        for item,checkpoint in MODELS.items():
            if args['method'] == item:
                state_dict = torch.load(checkpoint,weights_only=True)
                ssl_model.load_state_dict(state_dict, strict=False)
                print("* Successfully loading pretrained model.")
    

        # set fixed parameters
    # for name, param in ssl_model.wsi_embedder.named_parameters():
    #     # print(name)
    #     # if 'pre_attn' in name or 'attn.attention_a' in name or 'attn.attention_b' in name:
    #     if 'pre_attn' in name:
    #         param.requires_grad = False       

    # for name, param in ssl_model.module.rna_embedder.named_parameters():
    #     # print(name)
    #     if 'blocks.0' in name:
    #         param.requires_grad = False
        
    # for param in ssl_model.rna_reconstruction.parameters():
    #     param.requires_grad = True
        
    # only fine tune some fixed layers or only one layer

    
    wandb.watch(ssl_model)
    
    # set up optimizers
    print("* Setup optimizer...")
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, ssl_model.parameters()), lr=args["learning_rate"])
    
    # set up schedulers
    print("* Setup schedulers...")
    T_max = (args["epochs"] - args["warmup_epochs"]) * len(train_dataloader) if args["warmup"] else args["epochs"] * len(train_dataloader)
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=T_max,
        eta_min=args["end_learning_rate"]
    )
    
    if args["warmup"]:
        scheduler_warmup = LinearLR(
            optimizer, 
            start_factor=0.00001,
            total_iters=args["warmup_epochs"] * len(train_dataloader)
    )
    else:
        scheduler_warmup = None
    
    # set up losses
    print("* Setup losses...")
    loss_fn_interMod = InfoNCE(temperature=args["temperature"])
    loss_fn_rnaRecon = nn.MSELoss()
    loss_fn_intraMod = init_intra_wsi_loss_function(args) 

    # main training loop
    # best_rank = 0.
    # for epoch in range(begin_epoch+1, args["epochs"]+1):
        
    #     torch.cuda.empty_cache()
        
    #     print()
    #     print(f"Training for epoch {epoch}...")
    #     print()
        
    #     # train
    #     start = time.time()
    #     ep_loss, train_rank, ssl_model = train_loop(args, loss_fn_interMod, loss_fn_rnaRecon, loss_fn_intraMod, 
    #                                                             ssl_model, epoch, train_dataloader, optimizer, scheduler_warmup,
    #                                                             scheduler)
    #     end = time.time()
    #     wandb.log({"epoch": epoch, "train_loss": ep_loss, "train_rank": train_rank})
    #     write_dict_to_config_file({"epoch": epoch, "train_loss": ep_loss, "train_rank": train_rank},os.path.join(ROOT_SAVE_DIR, EXP_CODE,"log.json"))

    #     print()
    #     print(f"Done with epoch {epoch}")
    #     print(f"Total loss = {ep_loss}")
    #     print(f"Train rank = {train_rank}")
    #     print("Total time = {:.3f} seconds".format(end-start))


    #     # validation after each 10 epochs
    #     if epoch % 10 == 0 and epoch != 0:
    #         print('evaluating after {} iterations ...'.format(epoch))
    #         val_results = val_loop(CLF, ssl_model, val_dataloader)
    #         wandb.log(val_results)

    #     # Stop training based on rank of the training samples. Ok for TANGLE and Intra. 
    #     if STOPPING_CRITERIA == 'train_rank':
    #         if train_rank > best_rank:
    #             print('Better rank: {} --> {}. Saving model'.format(best_rank, train_rank))
    #             best_rank = train_rank
    #             torch.save(ssl_model.state_dict(), os.path.join(RESULTS_SAVE_PATH, "model.pt"))
    #             print()
    #             print('Reach the best rank and training is stopped.')
    #             print('Trianed Model is saved.')
    #             print()

    #     # Otherwise, stop after fixed number of training epochs. Ok for TANGLE-Rec. 
    #     elif epoch in args['save_epoch_list'] == True:
    #         epoch_dir = os.path.join(RESULTS_SAVE_PATH,str(epoch))
    #         os.mkdir(epoch_dir)
    #         torch.save(ssl_model.state_dict(), os.path.join(epoch_dir,"model_iterated.pt"))
    #         print('Trianed Model for {} times iteration is saved.'.format(epoch))
    #     else:
    #         torch.save(ssl_model.state_dict(), os.path.join(RESULTS_SAVE_PATH, "model.pt"))
    #         print()
    #         print('Train epoch reach maximum.')
    #         print('Trianed Model is saved.')
    #         print()
            
    print('testing ...')
    # testing

    # state_dict = torch.load(os.path.join(RESULTS_SAVE_PATH, "model.pt"),weights_only=True)
    # ssl_model.load_state_dict(state_dict, strict=True)
    


    
    total = []
    # auc,acc,f1,recall,pre,total = [],[],[],[],[],[]
    for i in range(30):
         test_result = test_loop(CLF,ssl_model,test_dataloader)
         wandb.log(test_result)
        # auc.append(test_result['auc'])
        # acc.append(test_result['acc'])
        # f1.append(test_result['f1'])
        # recall.append(test_result['recall'])
        # pre.append(test_result['pre'])
    #     total.append(test_result)
    # with open(os.path.join(args['save_root'], args['total_feature_path'].split("/")[-1]+".pt"),"wb") as f:
    #     pickle.dump(f)
    
    print("All results saved locally.")
    print()
    print("Done.")
    print()
            
