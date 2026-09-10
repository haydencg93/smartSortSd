import cv2
import torch
import time
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F

# Load 6-class model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 6)
model.load_state_dict(torch.load('smart_sort_mobilenetv2_6class.pth', map_location='cpu'))
model.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# TrashNet's alphabetical class order
material_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Iowa City Logic Mapping
iowa_city_rules = {
    'cardboard': 'Recycling',
    'metal': 'Recycling',
    'paper': 'Recycling',
    'plastic': 'Recycling',
    'glass': 'Trash',  # Excluded from standard bins
    'trash': 'Trash'
}

cap = cv2.VideoCapture(0)
print("Smart Sort Active. Center the item in the green box and press SPACE. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    box_size = int(min(h, w) * 0.6)
    x1 = (w - box_size) // 2
    y1 = (h - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    display_frame = frame.copy()
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow('Smart Sort - PC Prototype', display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 32: 
        start_t = time.time()

        roi = frame[y1:y2, x1:x2]
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess(rgb_roi).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probs, 0)

        latency = time.time() - start_t
        
        # Get the specific material and map it to the bin
        detected_material = material_classes[predicted_idx.item()]
        bin_decision = iowa_city_rules[detected_material]

        print(f"\n--- Classification Result ---")
        print(f"Detected Material: {detected_material.upper()} ({confidence.item()*100:.1f}%)")
        print(f"Iowa City Routing: {bin_decision.upper()}")
        
        print("Material Probabilities:")
        for i, material in enumerate(material_classes):
            print(f"  - {material}: {probs[i].item()*100:.1f}%")
            
        print(f"Latency: {latency:.4f}s")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()