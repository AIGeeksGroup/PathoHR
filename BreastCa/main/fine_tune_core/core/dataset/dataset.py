# --> General imports
import os
import numpy as np
import torch
import h5py
import pickle
import pandas as pd

# --> Torch imports 
import torch
from torch.utils.data import Dataset
import pdb


class TangleDataset(Dataset):
    def __init__(self, feats_dir, rna_dir,labels_dir, n_tokens, sampling_strategy="random"):
        """
        - feats_dir: str, directory where feat .pt files are stored
        - rna_dir: str, directory where rna_data .pt files are stored
        - n_tokens: int, number of tokens/patches to sample from all features
        - sampling_strategy: str, strategy to sample patches (only "random" available)
        """
        self.feats_dir = feats_dir
        self.rna_dir = rna_dir
        self.labels_dir = labels_dir
        self.n_tokens = n_tokens
        self.sampling_strategy = sampling_strategy
        
        # special operations for laoding the dir
        feature_dir = sorted(os.listdir(feats_dir),key=self.get_key)
        rna_dir = sorted(os.listdir(rna_dir),key=self.get_key)
        self.labels = list(pd.read_csv(self.labels_dir)['os_x'].values)
        # print(self.labels)
        # print(len(self.labels)) # 740, 92, 94
        
        # move the index in the begining of the file
        self.feat_slide_ids = [fname.split(".pt")[0] for fname in feature_dir if fname.endswith(".pt")]
        self.rna_ids = [fname.split(".pt")[0] for fname in rna_dir if fname.endswith(".pt")]
        
        # print(len(self.slide_ids)) 740,92,94
        # print(len(self.rna_ids))
        self.slide_ids_ = list(set(self.feat_slide_ids) | set(self.rna_ids))
    
    def get_key(self,string:str):
        return int(string.split('_')[0])
    
    def __len__(self):
        return len(self.feat_slide_ids)

    def __getitem__(self, idx):
        # slide_id = self.slide_ids[idx]
        slide_id = self.feat_slide_ids[idx]
        rna_id = self.rna_ids[idx]
        label = self.labels[idx]
        # print(label.shape) #[0]
        # print(label)
        # Load features and coords 
        patch_emb = torch.load(os.path.join(self.feats_dir, f"{slide_id}.pt"))
        # print('patch_emb:',patch_emb.shape)
        
        # - Avg patch embedding 
        patch_emb_avg = patch_emb.mean(dim=0)
        # print('patch_emb_avg:',patch_emb_avg)
        
        # Original 
        patch_indices = torch.randint(0, patch_emb.shape[0], (self.n_tokens,))
        patch_emb_ = patch_emb[patch_indices]
        # print('patch_emb_:',patch_emb.shape)

        # And an augmentation
        patch_indices_aug = torch.randint(0, patch_emb.size(0), (self.n_tokens,)).tolist() if patch_emb.shape[0] < self.n_tokens else torch.randperm(patch_emb.size(0))[:self.n_tokens].tolist()           
        patch_emb_aug = patch_emb[patch_indices_aug]
        # print('patch_emb_aug:',patch_emb_aug)

        # Load gene expression data 
        rna_data = torch.load(os.path.join(self.rna_dir, f"{rna_id}.pt"))
        # print('rna_data:',rna_data.shape)

        return patch_emb_, rna_data, label, patch_emb_aug, patch_emb_avg
    
    # deparcted
    def extract_patch_loop(self):
        patch_embeds = []
        for idx in self.feat_slide_ids:
            patch_emb = torch.load(os.path.join(self.feats_dir, f"{idx}.pt"))
            patch_indices = torch.randint(0, patch_emb.shape[0], (self.n_tokens,))
            patch_emb_ = patch_emb[patch_indices]
            patch_embeds.append(patch_emb_)
        return patch_embeds

def load_h5(h5_path):
    with h5py.File(h5_path, 'r') as hdf5_file:
        feats = hdf5_file['features'][:].squeeze()
    if isinstance(feats, np.ndarray):
        feats = torch.Tensor(feats)
    return feats


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


class SlideDataset(Dataset):
    def __init__(self, features_path, extension='.h5'):
        """
        Args:
            features_path (string): Directory with all the feature files.
        """
        self.features_path = features_path
        self.extension = extension
        self.slide_names = [x for x in os.listdir(features_path) if x.endswith(extension)]
        self.n_slides = len(self.slide_names)

    def __len__(self):
        return self.n_slides

    def __getitem__(self, index):
        slide_id = self.slide_names[index].replace(self.extension, '')
        feature_file = self.slide_names[index]
        feature_path = f"{self.features_path}/{feature_file}"
        if self.extension == '.pt':
            features = torch.load(feature_path)
        elif self.extension == '.h5':
            features = load_h5(feature_path)
        return features, slide_id


class FewShotClassificationDataset:
    """
    Dataset for few-shot classification in TCGA.
    Returns the slide embedding, binary label, and the slide ID.
    """

    def __init__(self, feature_folder_path, csv_path):
        self.feature_folder_path = feature_folder_path
        self.slide_embeddings = self.get_features()
        self.labels = pd.read_csv(csv_path)
        
    def get_features(self):
        if os.path.isfile(self.feature_folder_path):
            file = open(self.feature_folder_path, 'rb')
            obj = pickle.load(file)
            embeddings = obj['embeds']
        return embeddings 

    def get_few_shot_binary_datasets(self, k=None):
        
        if k is None: # Keep all samples
            k_neg_indices = np.where(self.labels == 0)[0].tolist()
            k_pos_indices = np.where(self.labels == 1)[0].tolist()
        else:
            k_neg_indices = np.random.sample(np.where(self.labels == 0)[0].tolist(), k) 
            k_pos_indices = np.random.sample(np.where(self.labels == 1)[0].tolist(), k) 

        neg_labels = self.labels[k_neg_indices]
        pos_labels = self.labels[k_pos_indices]

        neg_features = [self.slide_embeddings[idx] for idx in k_neg_indices]
        pos_features = [self.slide_embeddings[idx] for idx in k_pos_indices]
        data_dict = {}
        data_dict["features"] = np.concatenate((pos_features, neg_features))
        data_dict["binary_classes"] = np.concatenate((pos_labels, neg_labels))
        return data_dict


class featureonly_TangleDataset(Dataset):
    def __init__(self, feats_dir, labels_dir, n_tokens, sampling_strategy="random"):
        """
        - feats_dir: str, directory where feat .pt files are stored
        - rna_dir: str, directory where rna_data .pt files are stored
        - n_tokens: int, number of tokens/patches to sample from all features
        - sampling_strategy: str, strategy to sample patches (only "random" available)
        """
        self.feats_dir = feats_dir
        self.feats_dir_list = os.listdir(feats_dir)
        self.labels = pd.read_csv(labels_dir)
        self.n_tokens = n_tokens
        self.sampling_strategy = sampling_strategy
        
        # special operations for laoding the dir
        # feature_dir = sorted(os.listdir(feats_dir),key=self.get_key)
        # self.labels = list(pd.read_csv(self.labels_dir)['os'].values)
    
        # print(self.labels)
        # print(len(self.labels)) # 740, 92, 94
        align = []
        features = []
        # move the index in the begining of the file
        self.feat_slide_ids = [fname.split(".pt")[0] for fname in self.feats_dir_list if fname.endswith(".pt")]
        for file in self.feat_slide_ids:
            # print(file)
            if self.labels['slide_id'].isin([file]).any() == True:
                mathced = self.labels[self.labels['slide_id'] == file]
                features.append(file)
                align.append(mathced['os'].values[0])
            # else:
            #     print('Missing os labels for:',file)
        self.labels = align
        self.feat_slide_ids = features
        # print(len(self.feat_slide_ids),len(self.labels)) # 1046
    
    def get_key(self,string:str):
        return int(string.split('_')[0])
    
    def __len__(self):
        return len(self.feat_slide_ids)

    def __getitem__(self, idx):
        # print(idx)
        # slide_id = self.slide_ids[idx]
        slide_id = self.feat_slide_ids[idx]
        label = self.labels[idx]
        # print(label.shape) #[0]
        # print(label)
        # Load features and coords 
        patch_emb = torch.load(os.path.join(self.feats_dir, f"{slide_id}.pt"))
        # print('patch_emb:',patch_emb.shape)
        
        # - Avg patch embedding 
        patch_emb_avg = patch_emb.mean(dim=0)
        # print('patch_emb_avg:',patch_emb_avg)
        
        # Original 
        patch_indices = torch.randint(0, patch_emb.shape[0], (self.n_tokens,))
        patch_emb_ = patch_emb[patch_indices]
        # print('patch_emb_:',patch_emb.shape)

        # And an augmentation
        patch_indices_aug = torch.randint(0, patch_emb.size(0), (self.n_tokens,)).tolist() if patch_emb.shape[0] < self.n_tokens else torch.randperm(patch_emb.size(0))[:self.n_tokens].tolist()           
        patch_emb_aug = patch_emb[patch_indices_aug]
        # print('patch_emb_aug:',patch_emb_aug)



        return patch_emb_, label, patch_emb_aug, patch_emb_avg