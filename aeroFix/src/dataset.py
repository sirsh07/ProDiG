import json
import torch
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np

from easydict import EasyDict as edict
from geometry import get_mask_and_plucker, get_epipolar_mask_plucker



def process_pose(pose_path):
    
    pose = np.load(pose_path)
    return torch.from_numpy(pose).float()

def ref_process_pose(pose_path):
    
    pose = np.load(pose_path.replace("pixels", "ref").replace(".npy", "_K.npz"))
    return torch.from_numpy(pose["pose"]).float(), torch.from_numpy(pose["K"]).float()

def generate_pose_masks_old(pose1, pose2, k, height, width, swap_attn=False):
    
    
    def calc_fov_from_k(K, width, height, degrees=True):
        """
        Compute horizontal, vertical, and diagonal FOV from intrinsics K and image size.
        Works even if the principal point is not centered. Assumes pinhole (no fisheye).

        K: 3x3 intrinsic matrix [[fx, s, cx],
                                [0,  fy, cy],
                                [0,   0,  1]]
        width, height: image size in pixels
        """
        fx, s, cx = K[0, 0], K[0, 1], K[0, 2]
        fy, cy    = K[1, 1], K[1, 2]

        # Horizontal and vertical FOV allowing off-center principal point
        hfov = np.arctan((width - cx)/fx) + np.arctan(cx/fx)
        vfov = np.arctan((height - cy)/fy) + np.arctan(cy/fy)

        if degrees:
            hfov = np.degrees(hfov)
            vfov = np.degrees(vfov)

        return hfov, vfov
    
    cam_fov_h, cam_fov_v = calc_fov_from_k(k, width, height, degrees=False)
    cam_fov = (cam_fov_h + cam_fov_v) / 2.0
    noisy_frame = edict({"fov": cam_fov, "camera": pose1}); ref_frame = edict({"fov": cam_fov, "camera": pose2})
    
    latent_width, latent_height = width // 8, height // 8
    
    first_mask, second_mask, first_plucker, second_plucker = get_mask_and_plucker(noisy_frame, ref_frame, latent_width, dialate_mask=True)
    
    
    if swap_attn:
        return second_mask, first_mask , first_plucker, second_plucker
    else:
        return first_mask, second_mask , first_plucker, second_plucker
    


def generate_pose_masks(pose1, pose2, k, height, width, swap_attn=False):
    
    
    latent_width, latent_height = width // 8, height // 8
    
    first_mask, second_mask, first_plucker, second_plucker = get_epipolar_mask_plucker(pose1.unsqueeze(0), pose2.unsqueeze(0), k.unsqueeze(0), latent_height, latent_width)
    
    
    if swap_attn:
        return second_mask, first_mask , first_plucker, second_plucker
    else:
        return first_mask, second_mask , first_plucker, second_plucker
    



class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, height=576, width=1024, tokenizer=None, swap_attn=False):

        super().__init__()
        with open(dataset_path, "r") as f:
            self.data = json.load(f)[split]
        self.img_ids = list(self.data.keys())
        self.image_size = (height, width)
        self.tokenizer = tokenizer
        self.swap_attn = swap_attn

    def __len__(self):

        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]
        
        
        input_img = self.data[img_id]["image"]
        output_img = self.data[img_id]["target_image"]
        # NOTE (Sirsh)-(06-20-2025-16:42): Removed reference image handling
        ref_img = self.data[img_id]["ref_image"] if "ref_image" in self.data[img_id] else None
        # ref_img = None
        caption = self.data[img_id]["prompt"]
        
        if "pose" in self.data[img_id]:
            pose = process_pose(self.data[img_id]["pose"])
        else:
            pose = None
            
        if "ref_pose" in self.data[img_id]:
            ref_pose, ref_k = ref_process_pose(self.data[img_id]["ref_pose"])
        else:
            ref_pose = None
                        
        
        try:
            input_img = Image.open(input_img).convert("RGB")
            output_img = Image.open(output_img).convert("RGB")
        except:
            print("Error loading image:", input_img, output_img)
            return self.__getitem__(idx + 1)
        

        # img_t = F.to_tensor(img_t)
        # NOTE (Sirsh)-(06-20-2025-16:42): changed input_img to img_t
        img_t = F.to_tensor(input_img)
        img_t = F.resize(img_t, self.image_size)
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

        # output_t = F.to_tensor(output_t)
        # NOTE (Sirsh)-(06-20-2025-16:44): changed output_img to output_t
        output_t = F.to_tensor(output_img)
        output_t = F.resize(output_t, self.image_size)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        if (pose is not None) and (ref_pose is not None):
            
            _, ht, wt = img_t.shape
            
            first_mask, _, first_plucker, second_plucker =  generate_pose_masks(pose, ref_pose, ref_k, ht, wt, self.swap_attn)
            

        if ref_img is not None:
            ref_t = Image.open(ref_img).convert("RGB")
            ref_t = F.to_tensor(ref_t)
            ref_t = F.resize(ref_t, self.image_size)
            ref_t = F.normalize(ref_t, mean=[0.5], std=[0.5])
            try:
                img_t = torch.stack([img_t, ref_t], dim=0)
            except:
                import pdb; pdb.set_trace()
            output_t = torch.stack([output_t, ref_t], dim=0)            
        else:
            img_t = img_t.unsqueeze(0)
            output_t = output_t.unsqueeze(0)

        out = {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
        }
        
        if pose is not None:
            out["pose"] = pose
        if ref_pose is not None:
            out["ref_pose"] = ref_pose
            out["ref_K"] = ref_k
            out["first_mask"] = first_mask
            # out["second_mask"] = second_mask
            out["first_plucker"] = first_plucker
            out["second_plucker"] = second_plucker
        
        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            out["input_ids"] = input_ids

        return out
