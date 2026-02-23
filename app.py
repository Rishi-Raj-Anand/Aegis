import os
import cv2
import sys

# --- Import your Aegis modules ---
from Aegis.cloaking import AegisCloakingEngine
from Aegis.utils import show_comparison, save_image
from Aegis.FaceRedactor import FaceRedactor
from Aegis.TextRedactor import TextRedactor

# Global Configuration
TARGET_POOL = "assets/targets/"

def preview_and_save(input_path,result,F):
    print("Displaying comparison...")
    show_comparison(input_path, result)
    
    print("Saving result...")
    save_image(result,F)

def get_image_path_from_user():
    """Helper to get a valid image path from the user."""
    while True:
        path = input("\n[INPUT] Enter the path to your input image: ").strip().strip('"').strip("'")
        if os.path.exists(path) and os.path.isfile(path):
            return path
        print("❌ Error: File not found. Please try again.")

def task_redact_faces(input_path):
    """Redacting Faces"""
    print("\n--- FACE REDACTION ENGINE ---")    
    redactor = FaceRedactor()
    
    # Load and Detect
    img = redactor.load_image(input_path)
    boxes = redactor.get_detections(img)
    
    if len(boxes) == 0:
        print("⚠️ No faces detected in this image.")
        return

    # Show Preview 
    print("Displaying preview with box numbers...")
    redactor.plot_preview(img, boxes)
    
    # Get User Input
    box_input = input("Enter box numbers to blur (e.g., '1 2'): ")
    
    # Parse Input
    try:
        to_blur = [int(x) for x in box_input.replace(',', ' ').split()]
    except ValueError:
        print("❌ Invalid input. Skipping redaction.")
        return

    # Apply and Save
    result = redactor.apply_redaction(img, boxes, to_blur)

    return result
    


def task_redact_pii(input_path):
    """Redacting PII (Text)"""
    print("\n--- PII REDACTION ENGINE ---")
    
    t_redactor = TextRedactor()
    
    print("Scanning image for text...")
    # detect_text usually requires a file path
    text_detections = t_redactor.detect_text(input_path)
    
    if not text_detections:
        print("⚠️ No text detected.")
        return
        
    print(f"✅ Detected {len(text_detections)} text regions.")
    
    # Load image for processing 
    img = cv2.imread(input_path)
    
    # Apply Auto Redaction
    result = t_redactor.auto_redact(img, text_detections)
    
    return result

def task_cloaking(input_path):
    """Cloaking Image"""
    print("\n--- AI CLOAKING ENGINE ---")
    
    # Initialize Engine
    print("Initializing Aegis Cloaking Engine (Loading Models)...")
    cloaker = AegisCloakingEngine(target_pool_path=TARGET_POOL)
    
    # Run Optimization
    print("Generating Cloak... (This may take a moment)")
    cloaked_face = cloaker.generate_cloak(
        input_path=input_path,
        iterations=200,
        lr=0.005,
        epsilon=0.05
    )
    
    return cloaked_face

def main():
    """Main Application Loop"""
    while True:
        print("\n" + "="*40)
        print("            🛡️    AEGIS   🛡️")
        print("="*40)
        print("1. Redact Faces")
        print("2. Redact PII")
        print("3. Cloaking")
        print("4. Quit")
        print("-" * 40)
        
        choice = input("Select an option (1-4): ").strip()
        
        if choice == '1':
            input_path = get_image_path_from_user()
            final_image=task_redact_faces(input_path)
            preview_and_save(input_path,final_image,"F")
        elif choice == '2':
            input_path = get_image_path_from_user()
            final_image=task_redact_pii(input_path)
            preview_and_save(input_path,final_image,"P")
        elif choice == '3':
            input_path = get_image_path_from_user()
            final_image=task_cloaking(input_path)
            preview_and_save(input_path,final_image,"C")
        elif choice == '4':
            print("\nExiting Aegis...")
            break
        else:
            print("❌ Invalid option. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()