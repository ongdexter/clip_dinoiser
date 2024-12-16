import os
import glob
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
from hydra import compose, initialize
from .models.builder import build_model
from .segmentation.datasets.pascal_context import PascalContextDataset

PALETTE = list(PascalContextDataset.PALETTE)

def list_of_strings(arg):
    return arg.split(',')

class ClipDino():
    def __init__(self, clip_dim=512, pca_dim=12):
        cfg = "clip_dinoiser.yaml"
        checkpoint_path = "./checkpoints/last.pt"
        initialize(config_path="configs", version_base=None)
        self.cfg = compose(config_name=cfg)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompts = ['keyboard']
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
        self.clip_dim = clip_dim
        self.pca_dim = pca_dim
        self.U = None
        self.S = None
        self.V = None
        self.mean = None

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
    
    def fit_pca(self, feat):
        # b, c, h, w
        h, w = feat.shape[-2:]
        reshaped_tensor = feat.permute(0, 2, 3, 1).reshape(-1, self.clip_dim)
        
        self.mean = reshaped_tensor.mean(dim=0)
        centered_data = reshaped_tensor - self.mean
        
        self.U, self.S, self.V = torch.pca_lowrank(centered_data, q=self.pca_dim, center=False, niter=2)

    def clip_to_pca(self, feat):
        # 1, c, h, w
        feat = feat.squeeze()
        h, w = feat.shape[-2:]
        reshaped_tensor = feat.squeeze().permute(1, 2, 0).reshape(-1, self.clip_dim)
                
        # Compute mean and center the data
        self.mean = reshaped_tensor.mean(dim=0)
        centered_data = reshaped_tensor - self.mean

        # # Perform PCA using torch.pca_lowrank
        # self.U, self.S, self.V = torch.pca_lowrank(centered_data, q=self.pca_dim, center=False, niter=2)
        
        # Project the data onto the principal components
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        reduced_tensor = torch.matmul(centered_data, self.V[:, :self.pca_dim])
        end.record()
        torch.cuda.synchronize()
        print('projection:', start.elapsed_time(end))
        
        # Reshape back to image dimensions
        reduced_tensor = reduced_tensor.reshape(h, w, self.pca_dim)
        reduced_tensor = reduced_tensor.permute(2, 0, 1).unsqueeze(0)
        
        return reduced_tensor

    def pca_to_clip(self, reduced_feat):
        h, w = reduced_feat.shape[-2:]
        reduced_feat = reduced_feat.squeeze().permute(1, 2, 0).reshape(-1, self.pca_dim)
        
        # Invert the PCA transformation
        recovered_feat = torch.matmul(reduced_feat, self.V[:, :self.pca_dim].t())
        
        # Add back the mean
        recovered_feat += self.mean
        
        # Reshape back to image dimensions
        recovered_feat = recovered_feat.reshape(h, w, self.clip_dim)
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

if __name__ == '__main__':
    file_paths = sorted(glob.glob('/home/odexter/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/*.png'))
    
    clipdino = ClipDino()
    
    clip_feats = []
    for file_path in file_paths:
        # skip every other frame
        if file_paths.index(file_path) % 2 != 0:
            continue
        img = Image.open(file_path).convert('RGB')
        img = T.PILToTensor()(img).unsqueeze(0).to("cuda") / 255.
        clip_feat = clipdino.get_clipdino_features(img)
        clip_feats.append(clip_feat)
        
    clip_feats = torch.cat(clip_feats, dim=0)
    clipdino.fit_pca(clip_feats)
    
    for file_path in file_paths:
        img = Image.open(file_path).convert('RGB')
        img = T.PILToTensor()(img).unsqueeze(0).to("cuda") / 255.
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        print(img.shape)
        feat_pca = clipdino.get_pca_features(img)
        print(feat_pca.shape)
        feat_pca = F.interpolate(feat_pca, scale_factor=4, mode="bilinear", align_corners=False)
        feat_pca = F.interpolate(feat_pca, scale_factor=0.25, mode="bilinear", align_corners=False)
        print('feat_pca:', feat_pca.shape)
        relevancy = clipdino.get_relevancy_from_pca(feat_pca)
        
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end))
        
        # plt.imshow(relevancy[0][1].detach().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
        # plt.show()
