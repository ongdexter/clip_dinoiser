import os
from .models.builder import build_model
from hydra import compose, initialize, initialize_config_dir
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
import torch.nn.functional as F
import numpy as np
from operator import itemgetter
import torch
from .segmentation.datasets.pascal_context import PascalContextDataset
# from helpers.visualization import mask2rgb
from typing import List
import glob
from sklearn.decomposition import PCA
# from .models.gs_lerf_field import GaussianLERFField

PALETTE = list(PascalContextDataset.PALETTE)

def list_of_strings(arg):
    return arg.split(',')

class ClipDino():
    def __init__(self, feature_mode='pca', clip_dim=512, pca_dim=64):
        cfg="configs/clip_dinoiser.yaml"
        checkpoint_path = "checkpoints/last.pt"
        base_path = os.path.dirname(os.path.realpath(__file__))
        cfg = '/' + os.path.join(base_path, cfg)
        checkpoint_path = '/' + os.path.join(base_path, checkpoint_path)
        # initialize(config_path='configs', version_base=None)
        # self.cfg = compose(config_name=cfg)
        with initialize_config_dir(version_base=None, config_dir='/' + base_path + '/configs'):
            # Compose the configuration
            self.cfg = compose(config_name="clip_dinoiser.yaml")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompts = ['monitor']
        if len(self.prompts) == 1:
            self.prompts = ['background'] + self.prompts
        self.model = build_model(self.cfg.model, class_names=self.prompts)
        assert os.path.isfile(checkpoint_path), "Checkpoint file doesn't exist"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        self.model.to(self.device)
        if 'background' in self.prompts:
            self.model.apply_found = True
        else:
            self.model.apply_found = False
        self.feature_mode = feature_mode
        self.clip_dim = clip_dim
        self.pca_dim = pca_dim
        self.pca = PCA(n_components=self.pca_dim)
        # self.hash_field = GaussianLERFField()
        # self.hash_optimizer = torch.optim.AdamW(self.hash_field.parameters(), lr=1e-3)
        
    def process_image(self, img_tens):
        if len(img_tens.shape) == 3:
            img_tens = img_tens.unsqueeze(0)
        h, w = img_tens.shape[-2:]
        with torch.amp.autocast('cuda'):
            output = self.model(img_tens)
        output = F.interpolate(output, scale_factor=self.model.vit_patch_size, mode="bilinear",
                               align_corners=False)[..., :h, :w]
        return output
    
    def get_clipdino_features(self, img_tens):
        if len(img_tens.shape) == 3:
            img_tens = img_tens.unsqueeze(0)
        h, w = img_tens.shape[-2:]
        with torch.amp.autocast('cuda'):
            output = self.model.get_clipdino_features(img_tens)
        return output
    
    def clip_to_pca(self, feat):
        feat = feat.squeeze()
        h, w = feat.shape[-2:]
        reshaped_tensor = feat.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        reduced_tensor = self.pca.fit_transform(reshaped_tensor.reshape(-1, 512)).reshape(h, w, self.pca_dim)
        reduced_tensor = torch.tensor(reduced_tensor).to("cuda")
        reduced_tensor = reduced_tensor.permute(2, 0, 1).unsqueeze(0)
        
        return reduced_tensor
    
    def pca_to_clip(self, reduced_feat):
        h, w = reduced_feat.shape[-2:]
        reduced_feat = reduced_feat.squeeze().transpose(1, 2, 0)
        recovered_feat = self.pca.inverse_transform(reduced_feat.reshape(-1, self.pca_dim)).reshape(h, w, 512)
        recovered_feat = torch.tensor(recovered_feat).to("cuda")
        recovered_feat = recovered_feat.permute(2, 0, 1).unsqueeze(0)
        
        return recovered_feat
    
    def get_pca_features(self, img_tens):
        feat = self.get_clipdino_features(img_tens)
        feat_pca = self.clip_to_pca(feat)
        
        return feat_pca
    
    def get_relevancy_from_pca(self, feat_pca):
        feat = self.pca_to_clip(feat_pca)
        
        return self.model.get_relevancy(feat)
    
    def get_relevancy(self, feat):
        return self.model.get_relevancy(feat)
    
    def viz_pca_features(self, feat_pca):
        if len(feat_pca.shape) == 3:
            feat_pca = feat_pca.unsqueeze(0)
        feat_pca = F.interpolate(feat_pca, size=(self.fh, self.fw), mode="bilinear", align_corners=False)        
        relevancy_image = self.get_relevancy_from_pca(feat_pca.detach().cpu().numpy())[0][1]
        plt.imshow(relevancy_image.detach().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
        plt.title('target')
        plt.show()
        
    def viz_relevancy(self, relevancy):
        plt.imshow(relevancy.detach().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
        plt.title('target')
        plt.show()

    def process_features(self, img, viz=False):
        if self.feature_mode == 'hash':
            # get relevancy map
            feat_clip = self.get_clipdino_features(img.unsqueeze(0))
            
            h, w = img.shape[-2:]
            feat_clip = F.interpolate(feat_clip, size=(h, w), mode="bilinear", align_corners=False)
            
            return feat_clip
                        
        elif self.feature_mode == 'pca':
            # get relevancy map
            feat_pca = self.get_pca_features(img.unsqueeze(0))
            
            h, w = img.shape[-2:]
            self.fh, self.fw = feat_pca.shape[-2:]
            # feat_pca = F.interpolate(feat_pca, size=(h, w), mode="bilinear", align_corners=False)            
            
            if viz:
                self.viz_pca_features(feat_pca)
            
            return feat_pca
        
        elif self.feature_mode == 'pca-hash':
            # get relevancy map
            feat_pca = self.get_pca_features(img.unsqueeze(0))
            
            h, w = img.shape[-2:]
            self.fh, self.fw = feat_pca.shape[-2:]
            
            return feat_pca

if __name__ == '__main__':
    file_paths = glob.glob('/home/odexter/datasets/replica/room0/results/img')
    
    clipdino = ClipDino()
    for file_path in file_paths:
        img = Image.open(file_path).convert('RGB')
        img = T.PILToTensor()(img).unsqueeze(0).to("cuda") / 255.
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        feat_pca = clipdino.get_pca_features(img)
        print('feat_pca:', feat_pca.shape)
        relevancy = clipdino.get_relevancy_from_pca(feat_pca)
        
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end))
        
        # print(output.shape)
        # print(np.min(output[0][1].detach().cpu().numpy()), np.max(output[0][1].detach().cpu().numpy()))
        # plt.imshow(output[0][1].detach().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
        # plt.colorbar()
        # plt.show()
        # exit()