import torch
import PIL.Image as Image
import numpy as np


class DepthModel:
    def __init__(self, model_type, subtype=None):
        self.model_type = model_type
        self.subtype = subtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_type == 'ZoeDepth':
            repo = "isl-org/ZoeDepth"
            model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)
            zoe = model_zoe_n.to(self.device)
            zoe.eval()
            self.model = zoe
        elif model_type == 'MiDaS':
            assert subtype is not None
            midas = torch.hub.load("intel-isl/MiDaS", subtype)
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if subtype == "DPT_Large" or subtype == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            elif subtype == "DPT_BEiT_L_512":
                self.transform = midas_transforms.beit512_transform
            elif subtype == "DPT_Swin_L_384":
                self.transform = midas_transforms.swin384_transform
            elif subtype == 'DPT_SwinV2_T_256':
                self.transform = midas_transforms.swin256_transform
            elif subtype == 'DPT_LeViT_224':
                self.transform = midas_transforms.levit_transform
            else:
                self.transform = midas_transforms.small_transform
            midas.to(self.device)
            midas.eval()
            self.model = midas

    def infer(self, img: Image):
        depth = None
        if self.model_type == 'ZoeDepth':
            depth = self.model.infer_pil(img)
        if 'MiDaS' == self.model_type:
            img_np = np.array(img)
            input_batch = self.transform(img_np).to(self.device)
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_np.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

                depth = prediction.cpu().numpy()
        depth_min = depth.min()
        depth_max = depth.max()
        depth = (depth - depth_min) / (depth_max - depth_min)
        if self.model_type == 'ZoeDepth':
            depth = 1 - depth  # ZoeDepth produces black=close and white=far
        return depth
