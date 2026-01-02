"""
License Plate Redactor Module
Part of Aegis - Image Anonymization System

This module detects and redacts license plates in images using computer vision.
"""

import cv2
import numpy as np
from typing import Tuple, List
from pathlib import Path


class LicensePlateRedactor:
    """
    A class to detect and redact license plates in images.
    
    Supports multiple redaction methods:
    - blur: Gaussian blur
    - pixelate: Pixelation effect
    - black: Complete blackout
    """
    
    def __init__(self, method: str = 'blur'):
        """
        Initialize the License Plate Redactor.
        
        Args:
            method: Redaction method ('blur', 'pixelate', or 'black')
        """
        self.method = method
        
        # Load Haar Cascade for license plate detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
        self.plate_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.plate_cascade.empty():
            raise RuntimeError(f"Failed to load cascade classifier from {cascade_path}")
    
    def detect_plates(self, image: np.ndarray) -> np.ndarray:
        """
        Detect license plates in the image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Array of detected plate coordinates (x, y, w, h)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect plates with multiple scale factors for better coverage
        plates = self.plate_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(25, 25)
        )
        
        return plates
    
    def blur_region(self, image: np.ndarray, x: int, y: int, 
                   w: int, h: int, strength: int = 51) -> np.ndarray:
        """
        Apply Gaussian blur to a specific region.
        
        Args:
            image: Input image
            x, y, w, h: Region coordinates
            strength: Blur strength (must be odd)
            
        Returns:
            Image with blurred region
        """
        roi = image[y:y+h, x:x+w]
        
        # Ensure kernel size is odd
        if strength % 2 == 0:
            strength += 1
        
        blurred = cv2.GaussianBlur(roi, (strength, strength), 0)
        image[y:y+h, x:x+w] = blurred
        
        return image
    
    def pixelate_region(self, image: np.ndarray, x: int, y: int, 
                       w: int, h: int, pixel_size: int = 10) -> np.ndarray:
        """
        Apply pixelation effect to a specific region.
        
        Args:
            image: Input image
            x, y, w, h: Region coordinates
            pixel_size: Size of pixels in the pixelated region
            
        Returns:
            Image with pixelated region
        """
        roi = image[y:y+h, x:x+w]
        
        # Create pixelation effect by downsampling and upsampling
        small = cv2.resize(roi, (pixel_size, pixel_size), 
                          interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), 
                              interpolation=cv2.INTER_NEAREST)
        
        image[y:y+h, x:x+w] = pixelated
        
        return image
    
    def blackout_region(self, image: np.ndarray, x: int, y: int, 
                       w: int, h: int) -> np.ndarray:
        """
        Black out a specific region.
        
        Args:
            image: Input image
            x, y, w, h: Region coordinates
            
        Returns:
            Image with blacked out region
        """
        image[y:y+h, x:x+w] = 0
        return image
    
    def redact(self, image: np.ndarray, strength: int = 51, 
               show_detections: bool = False) -> Tuple[np.ndarray, int]:
        """
        Main function to detect and redact license plates.
        
        Args:
            image: Input image in BGR format
            strength: Redaction strength (blur strength or pixel size)
            show_detections: Whether to draw boxes around detections
            
        Returns:
            Tuple of (processed_image, number_of_plates_detected)
        """
        result = image.copy()
        plates = self.detect_plates(image)
        
        # Optionally draw detection boxes
        if show_detections:
            for (x, y, w, h) in plates:
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Apply redaction based on selected method
        for (x, y, w, h) in plates:
            if self.method == 'blur':
                result = self.blur_region(result, x, y, w, h, strength)
            elif self.method == 'pixelate':
                result = self.pixelate_region(result, x, y, w, h, strength)
            elif self.method == 'black':
                result = self.blackout_region(result, x, y, w, h)
        
        return result, len(plates)
    
    def process_image(self, input_path: str, output_path: str, 
                     strength: int = 51, show_detections: bool = False) -> int:
        """
        Process a single image file.
        
        Args:
            input_path: Path to input image
            output_path: Path to save processed image
            strength: Redaction strength
            show_detections: Whether to show detection boxes
            
        Returns:
            Number of plates detected
        """
        # Read image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not read image from {input_path}")
        
        # Process
        processed, num_plates = self.redact(image, strength, show_detections)
        
        # Save
        cv2.imwrite(output_path, processed)
        
        return num_plates
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         strength: int = 51) -> Tuple[int, int]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            strength: Redaction strength
            
        Returns:
            Tuple of (total_images_processed, total_plates_detected)
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Supported extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        total_plates = 0
        processed_count = 0
        
        # Process each image
        for file_path in Path(input_dir).iterdir():
            if file_path.suffix.lower() in extensions:
                try:
                    output_path = Path(output_dir) / f"redacted_{file_path.name}"
                    num_plates = self.process_image(
                        str(file_path),
                        str(output_path),
                        strength=strength
                    )
                    total_plates += num_plates
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        return processed_count, total_plates


def main():
    """Example usage of the LicensePlateRedactor."""
    
    # Create redactor with blur method
    redactor = LicensePlateRedactor(method='blur')
    
    # Process single image
    try:
        num_plates = redactor.process_image(
            'assets/targets/sample_car.jpg',
            'assets/targets/redacted_car.jpg',
            strength=51
        )
        print(f"Detected and redacted {num_plates} license plate(s)")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()