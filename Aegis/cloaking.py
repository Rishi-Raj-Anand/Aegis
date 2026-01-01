import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time

class AegisCloakingEngine:
    def __init__(self, target_pool_path=None):
        """
        Implementation of the Fawkes Image Cloaking System.
        
        Args:
            target_pool_path: Path to a directory containing images of other people. 
                              Used to find the 'most dissimilar'
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Aegis Cloaking Engine Initialized on {self.device} ---")

        # 1. Feature Extractor
        # use ResNet50 as a standard robust proxy.
        resnet = models.resnet50(pretrained=True).to(self.device)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).eval()
        
        # Freeze gradients for the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # 2. Preprocessing 
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Inverse transform for visualization/saving
        self.denormalize = transforms.Compose([
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                                 std=[1/0.229, 1/0.224, 1/0.225]),
        ])

        self.target_pool_path = target_pool_path

    def load_image_as_tensor(self, path):
        """Loads image and normalizes it for the model."""
        img = Image.open(path).convert('RGB')
        return self.preprocess(img).unsqueeze(0).to(self.device)

    def select_optimal_target(self, user_features):
        """
        Step 1 : Choosing a Target Class
        Scans the target_pool directory to find the face 'most dissimilar' 
        to the user's face in feature space.
        """
        if not self.target_pool_path or not os.path.exists(self.target_pool_path):
            print("No target pool provided. Using random noise target (High Privacy).")
            return None

        print("Searching for optimal target identity...")
        max_dist = -1.0
        best_target_path = None
        
        valid_exts = ('.jpg', '.jpeg', '.png')
        for filename in os.listdir(self.target_pool_path):
            if not filename.lower().endswith(valid_exts):
                continue
                
            path = os.path.join(self.target_pool_path, filename)
            try:
                # Load candidate target
                t_tensor = self.load_image_as_tensor(path)
                with torch.no_grad():
                    t_features = self.feature_extractor(t_tensor)
                
                # Calculate L2 distance [cite: 258]
                dist = torch.dist(user_features, t_features).item()
                
                if dist > max_dist:
                    max_dist = dist
                    best_target_path = path
            except Exception as e:
                continue
        
        if best_target_path:
            print(f"Target Selected: {os.path.basename(best_target_path)} (Distance: {max_dist:.4f})")
        return best_target_path

    def generate_cloak(self, input_path, manual_target_path=None, iterations=1000, lr=0.005, epsilon=0.05):
        """
        Step 2: Computing Per-image Cloaks[cite: 259].
        
        Args:
            iterations: Paper suggests 1000 iterations.
            epsilon: Budget for pixel change (e.g. 0.03-0.07).
            
        Returns:
            BGR numpy array ready for cv2.imwrite.
        """
        # A. Load Original Image
        original_tensor = self.load_image_as_tensor(input_path)
        
        # B. Extract User Features
        with torch.no_grad():
            user_features = self.feature_extractor(original_tensor)

        # C. Determine Target Features
        target_path = manual_target_path
        
        # If no manual target, find the best one automatically
        if not target_path:
            target_path = self.select_optimal_target(user_features)
            
        if target_path:
            target_tensor = self.load_image_as_tensor(target_path)
            with torch.no_grad():
                target_features = self.feature_extractor(target_tensor)
        else:
            # Fallback: Random features if no target images available
            target_features = torch.randn_like(user_features)

        # D. Initialize Delta (The Cloak)
        # We optimize 'delta', not the image itself.
        delta = torch.zeros_like(original_tensor, requires_grad=True).to(self.device)
        
        # E. Setup Optimizer
        # Adam optimizer 
        optimizer = optim.Adam([delta], lr=lr)
        mse_loss = nn.MSELoss()

        print(f"Starting Cloaking Process ({iterations} iters)...")
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # 1. Apply Delta to Image
            # Clamp delta to epsilon immediately to ensure constraints during forward pass
            clamped_delta = torch.clamp(delta, -epsilon, epsilon)
            cloaked_input = original_tensor + clamped_delta
            
            # 2. Extract Features of Cloaked Image
            current_features = self.feature_extractor(cloaked_input)
            
            # 3. Loss: Minimize distance to TARGET features 
            # We want the face to look like the Target, not the User.
            loss = mse_loss(current_features, target_features)
            
            loss.backward()
            optimizer.step()
            
            # 4. Project Delta (Constraint)
            # Ensure delta stays within valid epsilon range after update
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            
            if i % 100 == 0:
                print(f"Step {i}/{iterations} | Loss: {loss.item():.6f}")

        # F. Generate Final Image
        with torch.no_grad():
            # Apply final optimized delta
            final_delta = torch.clamp(delta, -epsilon, epsilon)
            final_tensor = original_tensor + final_delta
            
            # Denormalize back to [0,1] range for saving
            # Remove batch dim -> (C, H, W)
            final_tensor = self.denormalize(final_tensor.squeeze(0)) 
            
            # Clamp to ensure valid image range [0, 1]
            final_tensor = torch.clamp(final_tensor, 0, 1)
            
            # Convert to Numpy (H, W, C)
            final_image = final_tensor.permute(1, 2, 0).cpu().numpy()
            
            # Convert RGB to BGR 
            final_image = (final_image * 255).astype(np.uint8)
            final_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
            
            return final_bgr

