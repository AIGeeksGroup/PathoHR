import argparse
import wandb
import torch

def process_args():
    # parser = wandb.config

    parser = argparse.ArgumentParser(description='Configurations for TANGLE pretraining')

    #----> Tangle (BRCA or NSCLC) vs Tanglev2 (Pancancer)
    parser.add_argument('--study', type=str, default='brca', help='Study: brca, nsclc or pancancer')

    #-----> model args 
    parser.add_argument('--embedding_dim', type=int, default=1024, help='Size of the embedding space')
    parser.add_argument('--rna_encoder', type=str, default="mlp", help='MLP or Linear.')
    parser.add_argument('--sampling_strategy', type=str, default="random", help='How to draw patch embeddings.')
    parser.add_argument('--wsi_encoder', type=str, default="abmil", help='Type of MIL.')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of heads in ABMIL.')
    parser.add_argument('--hidden_dim', type=int, default=768, help='Internal dim of ABMIL.')
    parser.add_argument('--activation', type=str, default='softmax', help='Activation function used in ABMIL attention weight agg (sigmoid or softmax).')
    parser.add_argument('--mask_percentage', type=float, default=0.5, help='Percentage of n_tokens that is masked during Intra loss computation.')
    parser.add_argument('--adapt_heads',type= int, default=8,help='Head amount for adaptiveTokenManager.')
    
    #----> training args
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Tensor dtype. Defaults to bfloat16 for increased batch size.')
    parser.add_argument('--warmup', type=bool, default=True, help='If doing warmup.')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs.')
    parser.add_argument('--epochs', type=int, default=500, help='maximum number of epochs to train (default: 2)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--end_learning_rate', type=float, default=1e-8, help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1234, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--temperature', type=float, default=0.01, help='InfoNCE temperature.')
    parser.add_argument('--gpu_devices', type=list, default=[0,1], help='List of GPUs.')
    parser.add_argument('--intra_modality_mode_wsi', type=str, default='reconstruct_masked_emb', help='Type of Intra loss. Options are: reconstruct_avg_emb, reconstruct_masked_emb.')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--n_tokens', type=int, default=2048, help='Number of patches to sample during training.')
    parser.add_argument('--symmetric_cl', type=bool, default=True, help='If use symmetric contrastive objective.')
    # no need for name 
    parser.add_argument('--method', type=str, default='', help='Train recipe. Options are: tangle, tanglerec, intra.')

    parser.add_argument('--num_workers', type=int, default=0, help='number of cpu workers')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay.')
    parser.add_argument('--feature_type', type=str, default='uni_feats', help='What type of features are you using?')

    #---> data inference
    parser.add_argument('--save_root', type = str, default=r"/date2/zhang_h/D/BreastCa/TANGLE/checkpoints/brca_checkpoints_and_embeddings_fine_tune/pre_avg_results",help="Save root path")
    parser.add_argument('--pin_memory',type=bool,default=True,help='pin momery in dataloader.')# tome_without_res_new
    parser.add_argument('--total_feature_path',type=str,default=r"/date2/zhang_h/D/BreastCa/data/modified_features/pooled_attention",help='Upload the whole feature dir.')
    parser.add_argument('--total_rna_path',type=str,default=r"/home/BreastCa/data/rna/new/total",help='Upload the whole rna dir.')
    parser.add_argument('--total_label_path',type=str,default=r"/date2/zhang_h/D/BreastCa/overallsurvival/final_ordered.csv",help='Upload the whole label dir.')
    parser.add_argument('--train_patch_embeds',type=str,default=r'/home/BreastCa/data/embeddings/train_patch_embeds.pkl',help='patch embedding of train data.')
    # parser.add_argument('--val_patch_embeds',type=str,default=r'/home/data2/zhangaho/1/zhangh/BreastCa/data/embeddings/val_patch_embeds.pth',help='patch embedding of val data.')
    # parser.add_argument('--test_patch_embeds',type=str,default=r'/home/data2/zhangaho/1/zhangh/BreastCa/data/embeddings/test_patch_embeds.pth',help='patch embedding of test data.')
    
    #---> model inference 
    parser.add_argument('--pretrained', type=str, default=r'/date2/zhang_h/D/BreastCa/Pretrain/intra_brca_lr0.0001_epochs100_bs64_tokensize2048_temperature0.01_uni/model.pt', help='Path to dir with checkpoint.')
    parser.add_argument('--linear_probing',type=bool,default=False,help='Wether linear probing using LG.')
    parser.add_argument('--save_epoch_list',type=list, default=['100','500','1000','1500','2000'], help='Epoch points for saving the model.')

    args = parser.parse_args()

    return args