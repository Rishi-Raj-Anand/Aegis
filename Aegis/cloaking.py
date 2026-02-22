import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import os
from pytorch_msssim import ssim

class AegisCloakingEngine:
    def __init__(self, target_pool_path=None):
        """
        Implementation of the Fawkes Image Cloaking System.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Aegis Cloaking Engine Initialized on {self.device} ---")

        # 1. Feature Extractor (Proxy)
        resnet = models.resnet50(pretrained=True).to(self.device)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).eval()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # 2. Preprocessing 
        # Note: We separate normalization so we can calculate SSIM on the [0, 1] range images
        self.to_tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), # Scales to [0, 1]
        ])
        
        self.normalize_for_model = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

        self.target_pool_path = target_pool_path

    def load_image_as_tensor(self, path):
        img = Image.open(path).convert('RGB')
        return self.to_tensor(img).unsqueeze(0).to(self.device)

    def select_optimal_target_centroid(self, user_features):
        """
        Step 1: Choosing a Target Class
        Calculates centroids for classes (subdirectories) and finds the most dissimilar one.
        """
        if not self.target_pool_path or not os.path.exists(self.target_pool_path):
            print("No target pool provided. Using random noise target.")
            return None

        print("Calculating class centroids to find optimal target...")
        max_dist = -1.0
        best_target_class = None
        best_target_features = None
        
        # Iterate through subdirectories (each represents a different person/class)
        for class_name in os.listdir(self.target_pool_path):
            class_dir = os.path.join(self.target_pool_path, class_name)
            if not os.path.isdir(class_dir): continue

            class_features = []
            for filename in os.listdir(class_dir):
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')): continue
                path = os.path.join(class_dir, filename)
                
                try:
                    t_tensor = self.load_image_as_tensor(path)
                    normalized_t = self.normalize_for_model(t_tensor)
                    with torch.no_grad():
                        features = self.feature_extractor(normalized_t)
                        class_features.append(features)
                except Exception:
                    continue
            
            if not class_features: continue
            
            # Calculate centroid of the class
            centroid = torch.mean(torch.stack(class_features), dim=0)
            
            # Find class maximizing L2 distance from user
            dist = torch.dist(user_features, centroid).item()
            if dist > max_dist:
                max_dist = dist
                best_target_class = class_name
                best_target_features = centroid
                
        if best_target_class:
            print(f"Target Class Selected: {best_target_class} (Distance: {max_dist:.4f})")
            
        return best_target_features

    # --- Tanh Space Transformations ---
    def to_tanh_space(self, x, eps=1e-6):
        """Maps an image from [0, 1] to unbounded tanh space."""
        # Clamp to avoid infinity in arctanh
        x = torch.clamp(x, eps, 1.0 - eps)
        # Scale to [-1, 1]
        x = (x - 0.5) * 2.0
        return torch.atanh(x)

    def from_tanh_space(self, w):
        """Maps unbounded tanh space back to [0, 1] image space."""
        return (torch.tanh(w) + 1.0) / 2.0

    def generate_cloak(self, input_path, iterations=1000, lr=0.5, rho=0.007, penalty_lambda=1000.0):
        """
        Step 2: Computing Per-image Cloaks
        """
        original_tensor = self.load_image_as_tensor(input_path) # Shape: [1, 3, 224, 224], Range: [0, 1]
        
        with torch.no_grad():
            norm_orig = self.normalize_for_model(original_tensor)
            user_features = self.feature_extractor(norm_orig)

        # Get Target Centroid Features
        target_features = self.select_optimal_target_centroid(user_features)
        if target_features is None:
            target_features = torch.randn_like(user_features)

        # Convert original image to tanh space to ensure pixel bounds
        w_orig = self.to_tanh_space(original_tensor).detach()
        
        # We optimize delta in tanh space
        delta_w = torch.zeros_like(w_orig, requires_grad=True).to(self.device)
        
        # Adam Optimizer as specified in paper
        optimizer = optim.Adam([delta_w], lr=lr) 

        print(f"Starting Cloaking Optimization (DSSIM budget: {rho})...")
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # 1. Apply delta and map back to [0, 1] image space
            cloaked_input = self.from_tanh_space(w_orig + delta_w)
            
            # 2. Extract features of the cloaked image
            norm_cloaked = self.normalize_for_model(cloaked_input)
            current_features = self.feature_extractor(norm_cloaked)
            
            # 3. Feature Loss: Minimize L2 distance to Target Centroid
            feature_loss = torch.dist(current_features, target_features, p=2)
            
            # 4. DSSIM Penalty Method
            # Calculate SSIM (value between 0 and 1)
            current_ssim = ssim(cloaked_input, original_tensor, data_range=1.0, size_average=True)
            # Convert to DSSIM (Structural Dis-similarity)
            current_dssim = (1.0 - current_ssim) / 2.0
            
            # Penalty activates only if DSSIM exceeds our budget rho
            dssim_penalty = torch.clamp(current_dssim - rho, min=0.0)
            
            # Total Loss combining feature displacement and visual fidelity constraint
            loss = feature_loss + (penalty_lambda * dssim_penalty)
            
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f"Step {i}/{iterations} | Loss: {loss.item():.4f} | DSSIM: {current_dssim.item():.5f}")

        # Final Generation
        with torch.no_grad():
            final_tensor = self.from_tanh_space(w_orig + delta_w)
            final_tensor = torch.clamp(final_tensor.squeeze(0), 0, 1)
            final_image = final_tensor.permute(1, 2, 0).cpu().numpy()
            
            final_image = (final_image * 255).astype(np.uint8)
            final_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
            
            return final_bgr