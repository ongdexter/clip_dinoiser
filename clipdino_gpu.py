import os
import glob
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
from hydra import compose, initialize, initialize_config_dir
from .models.builder import build_model
from .segmentation.datasets.pascal_context import PascalContextDataset

PALETTE = list(PascalContextDataset.PALETTE)

def list_of_strings(arg):
    return arg.split(',')

class ClipDino():
    def __init__(self, feature_mode='pca', clip_dim=512, pca_dim=12, data_path='/home/odexter/Downloads/ipca_coco_outdoor_24.pt'):
        cfg="configs/clip_dinoiser.yaml"
        self.checkpoint_path = "checkpoints/last.pt"
        self.base_path = os.path.dirname(os.path.realpath(__file__))
        cfg = '/' + os.path.join(self.base_path, cfg)
        self.checkpoint_path = '/' + os.path.join(self.base_path, self.checkpoint_path)
        with initialize_config_dir(version_base=None, config_dir='/' + self.base_path + '/configs'):
            # Compose the configuration
            self.cfg = compose(config_name="clip_dinoiser.yaml")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.set_prompt(['monitor'])
        self.feature_mode = feature_mode
        self.clip_dim = clip_dim
        self.pca_dim = pca_dim
        self.U = None
        self.S = None
        self.V = None
        self.mean = None        
        
        # load ipca
        # data = torch.load('/bags/ipca/ipca_coco_outdoor_12.pt')
        data = torch.load(data_path)
        self.mean = torch.tensor(data['mean']).to("cuda").float()
        self.V = torch.tensor(data['V']).to("cuda").float()
        
    def set_prompt(self, prompt, background_prompt=['grass']):
        self.prompts = prompt
        if len(self.prompts) == 1:
            self.prompts = background_prompt + self.prompts
        self.model = build_model(self.cfg.model, class_names=self.prompts)
        assert os.path.isfile(self.checkpoint_path), "Checkpoint file doesn't exist"        
        self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        self.model.load_state_dict(self.checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        self.model.to(self.device)
        if 'background' in self.prompts:
            self.model.apply_found = True
        else:
            self.model.apply_found = False

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
        self.fh, self.fw = feat.shape[-2:]
        reshaped_tensor = feat.permute(0, 2, 3, 1).reshape(-1, self.clip_dim)
        
        self.mean = reshaped_tensor.mean(dim=0)
        centered_data = reshaped_tensor - self.mean
        
        # To obtain repeatable results, reset the seed for the pseudorandom number generator
        torch.manual_seed(0)
        self.U, self.S, self.V = torch.pca_lowrank(centered_data, q=self.pca_dim, center=False, niter=2)

    def clip_to_pca(self, feat):
        # 1, c, h, w
        feat = feat.squeeze()
        h, w = feat.shape[-2:]
        reshaped_tensor = feat.squeeze().permute(1, 2, 0).reshape(-1, self.clip_dim)
                
        # Compute mean and center the data
        # self.mean = reshaped_tensor.mean(dim=0)
        centered_data = reshaped_tensor - self.mean

        # # Perform PCA using torch.pca_lowrank
        # self.U, self.S, self.V = torch.pca_lowrank(centered_data, q=self.pca_dim, center=False, niter=2)
        
        # Project the data onto the principal components
        reduced_tensor = torch.matmul(centered_data.float(), self.V.float().T)
        
        # Reshape back to image dimensions
        reduced_tensor = reduced_tensor.reshape(h, w, self.pca_dim)
        reduced_tensor = reduced_tensor.permute(2, 0, 1).unsqueeze(0)
        
        return reduced_tensor

    def pca_to_clip(self, reduced_feat):
        h, w = reduced_feat.shape[-2:]
        reduced_feat = reduced_feat.squeeze().permute(1, 2, 0).reshape(-1, self.pca_dim)
        
        # Invert the PCA transformation
        recovered_feat = torch.matmul(reduced_feat, self.V)
        
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
    
    def viz_pca_features(self, feat_pca):
        if len(feat_pca.shape) == 3:
            feat_pca = feat_pca.unsqueeze(0)
        feat_pca = F.interpolate(feat_pca, size=(self.fh, self.fw), mode="bilinear", align_corners=False)        
        relevancy_image = self.get_relevancy_from_pca(feat_pca.detach())[0][1]
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
