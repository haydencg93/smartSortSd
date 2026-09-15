import cv2
import requests
import time
from pyzbar.pyzbar import decode, ZBarSymbol

# Iowa City Logic Mapping
iowa_city_rules = {
    'cardboard': 'Recycling',
    'metal': 'Recycling',
    'paper': 'Recycling',
    'plastic': 'Recycling',
    'glass': 'Trash',  # Excluded from standard bins
    'trash': 'Trash'
}

# Standardized keywords to map API text to our material classes
material_keywords = {
    'plastic': ['plastic', 'pet', 'hdpe', 'bottle'],
    'glass': ['glass', 'jar'],
    'metal': ['metal', 'aluminum', 'tin', 'can'],
    'cardboard': ['cardboard', 'box', 'corrugated'],
    'paper': ['paper', 'wrapper'],
}

def determine_material(text_data):
    """Scans API response text for material keywords."""
    if not text_data:
        return None
    
    text_data = text_data.lower()
    for material, keywords in material_keywords.items():
        if any(keyword in text_data for keyword in keywords):
            return material
    return None

def lookup_barcode(barcode_data):
    """Queries Open Food Facts first, falls back to UPCitemdb."""
    
    # Add this line to initialize the variable with a default value
    found_name = "Unknown Item"
    
    # Pad 12-digit UPCs to 13 digits (EAN-13) for Open Food Facts compatibility
    off_barcode = barcode_data.zfill(13) if len(barcode_data) == 12 else barcode_data
    
    # Open Food Facts requires a User-Agent to prevent connection blocking
    headers = {
        'User-Agent': 'SmartSortRecyclingAssistant - Python - Version 1.0'
    }
    
    # 1. Open Food Facts API (v2)
    off_url = f"https://world.openfoodfacts.org/api/v2/product/{off_barcode}.json"
    try:
        response = requests.get(off_url, headers=headers, timeout=2.5)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 1:
                product = data.get('product', {})
                name = product.get('product_name', 'Unknown Item')
                
                # Broaden search to include packaging, categories, and serving size
                packaging_tags = " ".join(product.get('packaging_tags', []))
                packaging_text = str(product.get('packaging', ''))
                categories = " ".join(product.get('categories_tags', []))
                serving_size = str(product.get('serving_size', ''))
                
                # Combine all text fields to search for material keywords
                combined_text = f"{packaging_tags} {packaging_text} {categories} {serving_size}"
                material = determine_material(combined_text)
                
                if material:
                    return name, material
                
                # If OFF finds the item but no material, hold the name for fallback
                found_name = name
            else:
                found_name = "Unknown Item"
    except requests.exceptions.RequestException:
        found_name = "Unknown Item"
        
    # 2. UPCitemdb API (Free Trial Endpoint)
    upc_url = f"https://api.upcitemdb.com/prod/trial/lookup?upc={barcode_data}"
    try:
        response = requests.get(upc_url, headers=headers, timeout=2.5)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 'OK' and len(data.get('items', [])) > 0:
                item = data['items'][0]
                name = item.get('title', found_name) 
                
                search_text = f"{name} {item.get('description', '')}"
                material = determine_material(search_text)
                
                if material:
                    return name, material
                else:
                    return name, "unknown" 
    except requests.exceptions.RequestException:
        pass
        
    if found_name != "Unknown Item":
        return found_name, "unknown"
        
    return None, None

def main():
    cap = cv2.VideoCapture(0)
    print("Smart Sort Barcode Scanner Active. Hold a barcode up to the camera. Press 'q' to quit.")

    # Dictionary to debounce API calls
    last_scanned = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        barcodes = decode(frame, symbols=[
            ZBarSymbol.UPCA, 
            ZBarSymbol.UPCE, 
            ZBarSymbol.EAN13, 
            ZBarSymbol.EAN8, 
            ZBarSymbol.CODE128, 
            ZBarSymbol.CODE39
        ])

        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            barcode_data = barcode.data.decode("utf-8")
            current_time = time.time()

            # Debounce: Only process if we haven't scanned this code in the last 4 seconds
            if barcode_data not in last_scanned or (current_time - last_scanned[barcode_data]) > 4:
                print(f"\n[{time.strftime('%H:%M:%S')}] SCAN: {barcode_data}. Querying APIs...")
                
                start_t = time.time()
                name, material = lookup_barcode(barcode_data)
                latency = time.time() - start_t
                
                if material and material != "unknown":
                    decision = iowa_city_rules.get(material, "Trash")
                    
                    text = f"{name[:15]} - {decision.upper()} ->"
                    color = (255, 0, 0) if decision == 'Recycling' else (0, 0, 255)
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    print(f"Item: {name} | Material: {material} | Routing: {decision}")
                    print(f"Latency: {latency:.2f}s")
                
                else:
                    # Item not found or material unknown -> Trigger secondary vision path
                    text = "Not found. Place item for camera."
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                    
                    print(f"Lookup failed or material unknown (Item: {name}). Prompting Image Classification.")
                    print(f"Latency: {latency:.2f}s")
                
                last_scanned[barcode_data] = current_time

        cv2.imshow("Smart Sort - API Barcode Prototype", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()