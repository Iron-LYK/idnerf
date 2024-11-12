import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class latent_inference(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 **kwargs) -> None:
        super(latent_inference, self).__init__()
        self.latent_dim = latent_dim
        self.mix_1  = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=16,
                                kernel_size= 3, stride= 2, padding  = 1),
                        nn.BatchNorm2d(16),
                        nn.LeakyReLU(),
                        nn.Conv2d(16, out_channels=32,
                                kernel_size= 3, stride= 1, padding  = 1),
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(),        
                        nn.Conv2d(32, out_channels=32,
                                kernel_size= 3, stride= 1, padding  = 1),
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(),                                                                 
                        )
        self.mix_2  = nn.Sequential(
                        nn.Conv2d(96, out_channels=64,
                                kernel_size= 3, stride= 2, padding  = 1),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(),
                        nn.Conv2d(64, out_channels=64,
                                kernel_size= 3, stride= 2, padding  = 1),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(),        
                        nn.Conv2d(64, out_channels=32,
                                kernel_size= 3, stride= 1, padding  = 1),
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(),                                                                 
                        )        
        self.mix_3  = nn.Sequential(
                        nn.Conv2d(38, out_channels=32,
                                kernel_size= 3, stride= 2, padding  = 1),
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(),
                        nn.Conv2d(32, out_channels=32,
                                kernel_size= 3, stride= 2, padding  = 1),
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(),    
                        nn.Conv2d(32, out_channels=16,
                                kernel_size= 3, stride= 2, padding  = 1),
                        nn.BatchNorm2d(16),
                        nn.LeakyReLU(),        
                        nn.Conv2d(16, out_channels=16,
                                kernel_size= 3, stride= 1, padding  = 1),
                        nn.BatchNorm2d(16),
                        nn.LeakyReLU(),                                                                 
                        )
        self.final_layer = nn.Linear(16, self.latent_dim)
        
    def encode_input(self, input: torch.tensor):
        
        list = []
        for i in range(input.shape[0]):
            x_temp = input[i].unsqueeze(0)
            x_temp = self.mix_1(x_temp)
            list.append(x_temp)
        x = torch.cat(list,dim=1)
        x = self.mix_2(x)  

        return x
    
    def encode_tv(self, input: torch.tensor):
        x = self.mix_3(input)   
        return x   
    
    def normalize_images(self, images):
        shape = [*[1]*(images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(*shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(*shape).to(images.device)

        return (images - mean) / std
       
    def forward(self, imgs, camera, tv_camera, **kwargs):
        imgs = self.normalize_images(imgs).squeeze()
        device = imgs.device
        _, _, H, W = imgs.shape

        input_ray_images = []    
        for view_idx in range(camera['intrinsics'].shape[1]):
            K = camera['intrinsics'][:,view_idx].squeeze().to(device)  
            c2w = camera['extrinsics'][:,view_idx].squeeze().to(device) 
            input_ray_image = self.generate_ray_image(H, W, K, c2w)
            input_ray_images.append(input_ray_image)
        input_ray_images = torch.stack(input_ray_images, dim=0)
        x = torch.cat((imgs, input_ray_images),dim=1)
        latents_global = self.encode_input(x)

        # add tv_pose
        K = tv_camera['intrinsics'].squeeze().to(device)    # 3, 3
        c2w = tv_camera['extrinsics'].squeeze().to(device)  # 3, 4
        tv_ray_image = self.generate_ray_image(H, W, K, c2w)
        latents_tv = F.interpolate(latents_global, (H, W), mode='bilinear', align_corners=False)
        latents_tv = torch.cat((latents_tv, tv_ray_image.unsqueeze(0)),dim=1)    
        latents_tv = self.encode_tv(latents_tv)

        b,c,h,w = latents_tv.shape
        latents_tv = latents_tv.view(b,c,-1).permute(0,2,1)
        latents_tv = self.final_layer(latents_tv)
        latents_tv = latents_tv.permute(0,2,1).view(b, self.latent_dim, h, w)
        latents_tv = F.interpolate(latents_tv, (64, 64), mode='bilinear', align_corners=False)
        
        return latents_tv, input_ray_images

    def generate_ray_image(self, H, W, K, c2w):
        '''
        Params:
            H: height of the input image
            W: weight of the input image
            K: [3, 3] tensor, camera intrinsics matrix
            c2w: [3, 4] tensor, camera extrinsics matrix
        Return:
            ray_image: [6, H, W] tensor        
        '''
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H)) 
        i = i.t().to(K.device)
        j = j.t().to(K.device)

        # Calculate ray directions in camera frame
        dirs = torch.stack([
            (i-K[0][2])/K[0][0],    # x component
            -(j-K[1][2])/K[1][1],   # y component (negated)
            -torch.ones_like(i)     # z component, fixed to -1
            ], -1)

        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3,-1].expand(rays_d.shape)    # HW3
        
        rays_o = rays_o.permute(2,0,1) # [H W 3] -> [3 H W]
        rays_d = rays_d.permute(2,0,1) # [H W 3] -> [3 H W]
        ray_image = torch.cat((rays_o,rays_d),dim=0)  # [6 H W]
        
        return ray_image  