import os
from .models.builder import build_model
from hydra import compose, initialize
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

PALETTE = list(PascalContextDataset.PALETTE)

def list_of_strings(arg):
    return arg.split(',')

class ClipDino():
    def __init__(self):
        cfg = "clip_dinoiser.yaml"
        checkpoint_path = "clip_dinoiser/checkpoints/last.pt"
        initialize(config_path="configs", version_base=None)
        self.cfg = compose(config_name=cfg)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompts = ['exit the room']
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

    def process_image(self, img_tens):
        if len(img_tens.shape) == 3:
            img_tens = img_tens.unsqueeze(0)
        h, w = img_tens.shape[-2:]
        with torch.cuda.amp.autocast():
            output = self.model(img_tens)
        output = F.interpolate(output, scale_factor=self.model.vit_patch_size, mode="bilinear",
                               align_corners=False)[..., :h, :w]
        return output

if __name__ == '__main__':
    file_paths = glob.glob('/home/odexter/datasets/replica/room0/results/frame*.jpg')
    
    clipdino = ClipDino()
    for file_path in file_paths:
        img = Image.open(file_path).convert('RGB')
        img = T.PILToTensor()(img).unsqueeze(0).to("cuda") / 255.
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        output = clipdino.process_image(img)
        
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end))
        
        # print(output.shape)
        # print(np.min(output[0][1].detach().cpu().numpy()), np.max(output[0][1].detach().cpu().numpy()))
        # plt.imshow(output[0][1].detach().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
        # plt.colorbar()
        # plt.show()
        # exit()