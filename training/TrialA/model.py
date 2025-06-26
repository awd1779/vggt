# model.py
#
# Defines the custom Monocular2Map architecture, including the ObjectEncoder.

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import os
import sys

# Assume the vggt library is in the python path, managed by the calling script.
from vggt.models.vggt import VGGT

class ObjectEncoder(nn.Module):
    """A simple CNN to encode masked object image regions into single tokens."""
    def __init__(self, output_dim=1024):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        features = self.encoder(x).flatten(1)
        return self.projection(features)


class Monocular2Map(nn.Module):
    """
    The final Monocular2Map architecture. It integrates a VGGT backbone for geometry,
    an ObjectEncoder for semantics, and a classifier head.
    """
    def __init__(self, num_classes, embed_dim=1024):
        super().__init__()
        self.backbone = VGGT()
        self.object_encoder = ObjectEncoder(output_dim=embed_dim)
        self.semantic_classifier = nn.Linear(embed_dim, num_classes)
        
        self.object_transform = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, images, masks=None):
        """
        Implements the final hybrid forward pass.
        The VGGT backbone expects an un-batched sequence (S, C, H, W).
        """
        # The VGGT backbone takes an un-batched sequence and returns a batched prediction
        geom_predictions = self.backbone(images)

        if masks is None:
            return geom_predictions, None
        
        object_tokens_list = []
        # The masks tensor is also an un-batched sequence from the dataset
        S, O, H, W = masks.shape
        
        for i in range(S):
            img = images[i]
            img_masks = masks[i]
            
            object_crops = []
            for j in range(O):
                mask = img_masks[j]
                if not mask.any(): continue

                rows, cols = torch.any(mask, dim=1), torch.any(mask, dim=0)
                if not torch.any(rows) or not torch.any(cols): continue
                
                rmin, rmax = torch.where(rows)[0][[0, -1]]
                cmin, cmax = torch.where(cols)[0][[0, -1]]

                if rmax > rmin and cmax > cmin:
                    crop = img.unsqueeze(0)[:, :, rmin:rmax+1, cmin:cmax+1]
                    resized_crop = self.object_transform(crop)
                    object_crops.append(resized_crop.squeeze(0))
            
            if object_crops:
                object_crops_batch = torch.stack(object_crops, dim=0)
                tokens = self.object_encoder(object_crops_batch)
                object_tokens_list.append(tokens)

        if not object_tokens_list:
            return geom_predictions, None

        all_object_tokens = torch.cat(object_tokens_list, dim=0)
        semantic_logits = self.semantic_classifier(all_object_tokens)
        
        return geom_predictions, semantic_logits
