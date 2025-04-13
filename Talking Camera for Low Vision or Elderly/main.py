import sys
import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from collections import OrderedDict

# Add the CRAFT-pytorch repository folder to the Python path
craft_repo_path = os.path.join(os.path.dirname(__file__), 'CRAFT-pytorch')
if craft_repo_path not in sys.path:
    sys.path.append(craft_repo_path)

# Add the crnn.pytorch folder to the Python path
crnn_repo_path = os.path.join(os.path.dirname(__file__), 'crnn.pytorch')
if crnn_repo_path not in sys.path:
    sys.path.append(crnn_repo_path)

def remove_module_prefix(state_dict):
    """
    Remove 'module.' prefix from state dict keys if present.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    return new_state_dict

# --- Load the Pre-trained CRAFT Model ---
def load_craft_model(model_path='craft_refiner_CTW1500.pth', device='cpu'):
    try:
        sys.path.insert(0, craft_repo_path)
        from craft import CRAFT
    except ImportError:
        from craft import CRAFT

    model = CRAFT()
    print(f"Loading CRAFT model from {model_path}")
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = remove_module_prefix(state_dict)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        print("CRAFT model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading CRAFT model: {str(e)}")
        raise

# --- Load the Pre-trained CRNN Model ---
def load_crnn_model(model_path='crnn.pth', device='cpu'):
    try:
        from crnn.pytorch.models.crnn import CRNN
    except ImportError:
        try:
            from models.crnn import CRNN
        except ImportError:
            from crnn import CRNN
    
    print(f"Loading CRNN model from {model_path}")
    
    try:
        model = CRNN(imgH=32, nc=1, nclass=37, nh=256)
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = remove_module_prefix(state_dict)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        print("CRNN model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading CRNN model: {str(e)}")
        raise

# --- Preprocessing Functions ---
def preprocess_for_craft(image, target_size=1280):
    """
    Resize image to target_size while preserving aspect ratio.
    Convert image to tensor and normalize as required by the CRAFT model.
    """
    h, w, _ = image.shape
    ratio = min(target_size / h, target_size / w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    resized = cv2.resize(image, (new_w, new_h))
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor_img = transform(resized_rgb).unsqueeze(0)
    return tensor_img, ratio

def preprocess_for_crnn(roi):
    """
    Preprocess the cropped region for CRNN recognition.
    Convert the ROI to grayscale, resize to expected input size, and normalize.
    """
    if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
        print("Warning: Empty ROI received")
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w < 8:
        padded = cv2.copyMakeBorder(gray, 0, 0, 0, 8 - w, cv2.BORDER_CONSTANT, value=255)
        gray = padded
        h, w = gray.shape
    new_h = 32
    scale = new_h / h
    new_w = max(int(w * scale), 8)
    resized = cv2.resize(gray, (new_w, new_h))
    norm_img = resized.astype(np.float32) / 255.0
    norm_img = (norm_img - 0.5) / 0.5
    tensor_img = torch.FloatTensor(1, 1, new_h, new_w)
    tensor_img[0, 0, :, :] = torch.from_numpy(norm_img)
    return tensor_img

# --- Inference Functions ---
def detect_text_regions(craft_model, image, device='cpu', text_threshold=0.1, link_threshold=0.4, low_text=0.3):
    """
    Detect text regions using the CRAFT model.
    Here we assume the model output is downsampled by a factor of 2.
    Debug information is printed.
    """
    tensor_img, ratio = preprocess_for_craft(image)
    tensor_img = tensor_img.to(device)
    with torch.no_grad():
        y, _ = craft_model(tensor_img)
    # Assume downsampling factor = 2 (adjust if needed)
    factor = 2
    H_out = tensor_img.shape[2] // factor
    W_out = tensor_img.shape[3] // factor
    raw_output = y[0].view(H_out, W_out, 2)
    print(f"CRAFT raw output shape (after reshape): {raw_output.shape}")
    print(f"CRAFT raw output stats -> min: {raw_output.min().item()}, max: {raw_output.max().item()}, mean: {raw_output.mean().item()}")
    
    # Extract text score channel and apply sigmoid
    score_text = torch.sigmoid(raw_output[..., 0]).cpu().numpy()
    print(f"After sigmoid -> min: {score_text.min()}, max: {score_text.max()}")
    
    # Normalize the score map to [0,1]
    score_norm = (score_text - score_text.min()) / (score_text.max() - score_text.min() + 1e-6)
    print(f"Normalized score map -> min: {score_norm.min()}, max: {score_norm.max()}")
    
    binary_map = (score_norm > text_threshold).astype(np.uint8) * 255
    cv2.imshow("Binary Map", cv2.resize(binary_map, (binary_map.shape[1]*2, binary_map.shape[0]*2)))
    
    kernel = np.ones((3, 3), np.uint8)
    binary_map = cv2.dilate(binary_map, kernel, iterations=1)
    
    contours, _ = cv2.findContours(binary_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        rect = cv2.boundingRect(contour)
        x, y, w, h = rect
        # Scale back to original image coordinates
        # Multiply by factor then adjust with ratio
        x = int(x * factor / ratio)
        y = int(y * factor / ratio)
        w = int(w * factor / ratio)
        h = int(h * factor / ratio)
        if w > 5 and h > 5:
            boxes.append((x, y, w, h))
    print(f"Found {len(boxes)} text regions")
    return boxes

def recognize_text(crnn_model, roi, device='cpu'):
    """
    Recognize text in the given ROI using the CRNN model.
    Debug information is printed.
    """
    if roi is None or roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
        return ""
    tensor_roi = preprocess_for_crnn(roi)
    if tensor_roi is None:
        return ""
    tensor_roi = tensor_roi.to(device)
    with torch.no_grad():
        preds = crnn_model(tensor_roi)
    print(f"CRNN raw predictions shape: {preds.shape}")
    print(f"CRNN predictions stats -> min: {preds.min().item()}, max: {preds.max().item()}, mean: {preds.mean().item()}")
    text = greedy_decode(preds)
    return text

def greedy_decode(preds):
    """
    A greedy decoder for CRNN output.
    """
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    chars = []
    prev = -1
    mapping = {1:'A', 2:'B', 3:'C', 4:'D', 5:'E', 6:'F', 7:'G', 8:'H', 9:'I', 10:'J',
               11:'K', 12:'L', 13:'M', 14:'N', 15:'O', 16:'P', 17:'Q', 18:'R', 19:'S',
               20:'T', 21:'U', 22:'V', 23:'W', 24:'X', 25:'Y', 26:'Z',
               27:'0', 28:'1', 29:'2', 30:'3', 31:'4', 32:'5', 33:'6', 34:'7', 35:'8', 36:'9'}
    for i in range(preds.size(0)):
        if preds[i] != prev and preds[i] != 0:
            if preds[i].item() in mapping:
                chars.append(mapping[preds[i].item()])
        prev = preds[i]
    return ''.join(chars)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    craft_model_path = os.path.join(script_dir, 'craft_refiner_CTW1500.pth')
    crnn_model_path = os.path.join(script_dir, 'crnn.pth')
    print(f"Loading models from: {script_dir}")
    craft_model = load_craft_model(model_path=craft_model_path, device=device)
    crnn_model = load_crnn_model(model_path=crnn_model_path, device=device)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return
    
    print("Camera opened successfully.")
    print("Press 'c' to capture an image, 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        cv2.imshow('Live Feed - Press c to capture', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            print("Capturing image...")
            captured_image = frame.copy()
            print("Detecting text regions...")
            boxes = detect_text_regions(craft_model, captured_image, device=device)
            vis_image = captured_image.copy()
            full_text = ""
            for i, (x, y, w, h) in enumerate(boxes):
                x = max(0, x)
                y = max(0, y)
                w = min(w, captured_image.shape[1] - x)
                h = min(h, captured_image.shape[0] - y)
                roi = captured_image[y:y+h, x:x+w]
                if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                    continue
                print(f"Recognizing text in region {i+1}...")
                recognized = recognize_text(crnn_model, roi, device=device)
                print(f"Region {i+1} raw recognized text: {recognized}")
                if recognized:
                    full_text += recognized + "\n"
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(vis_image, recognized, (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow('Detected Text Regions', vis_image)
            print("\nFull Recognized Text:")
            print(full_text)
            if not full_text:
                print("No text was recognized in the image.")
        elif key == ord('q'):
            print("Quitting application...")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
