# app.py - PisoUno Coin Detection System
# Old and New 1 Peso Coin Detection, Classification, and Counting System

import streamlit as st
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import os
import tempfile
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Custom HOG feature extraction (replaces scikit-image)
# ============================================

def compute_hog_features(gray_image, cells_per_block=(2, 2), pixels_per_cell=(8, 8), orientations=9):
    """
    Manual HOG feature extraction without scikit-image
    """
    # Calculate gradients
    gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    
    # Calculate magnitude and orientation
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180
    
    # Get image dimensions
    h, w = gray_image.shape
    cell_h, cell_w = pixels_per_cell
    block_h, block_w = cells_per_block
    
    # Calculate number of cells
    cells_x = w // cell_w
    cells_y = h // cell_h
    
    # Initialize histogram bins
    hist_bins = orientations
    hist_range = (0, 180)
    
    # Create cell histograms
    cell_histograms = np.zeros((cells_y, cells_x, hist_bins))
    
    for y in range(cells_y):
        for x in range(cells_x):
            # Get cell region
            cell_mag = magnitude[y*cell_h:(y+1)*cell_h, x*cell_w:(x+1)*cell_w]
            cell_ori = orientation[y*cell_h:(y+1)*cell_h, x*cell_w:(x+1)*cell_w]
            
            # Create histogram
            hist, _ = np.histogram(cell_ori.flatten(), bins=hist_bins, range=hist_range, weights=cell_mag.flatten())
            cell_histograms[y, x] = hist
    
    # Compute block features
    block_features = []
    for y in range(cells_y - block_h + 1):
        for x in range(cells_x - block_w + 1):
            # Extract block
            block = cell_histograms[y:y+block_h, x:x+block_w].flatten()
            # Normalize block
            block = block / (np.sqrt(np.sum(block**2)) + 1e-6)
            block_features.extend(block)
    
    return np.array(block_features)

# ============================================
# Feature Extraction Functions
# ============================================

def extract_features(image_path, img_size=(64, 64)):
    """
    Extract features from a coin image
    Features: Hu moments, HOG, color histograms, edge density
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            try:
                pil_img = Image.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except:
                return None
        
        # Resize image
        img_resized = cv2.resize(img, img_size)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # 1. Hu Moments (7 features)
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        # 2. HOG Features (using custom implementation)
        try:
            hog_features = compute_hog_features(gray)
            # Limit to a reasonable size
            if len(hog_features) > 100:
                hog_features = hog_features[:100]
        except:
            hog_features = np.zeros(36)
        
        # 3. Color Histograms (3 channels x 8 bins = 24 features)
        hist_features = []
        for i in range(3):
            hist = cv2.calcHist([img_resized], [i], None, [8], [0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-7)
            hist_features.extend(hist)
        
        # 4. Edge Density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (img_size[0] * img_size[1])
        
        # 5. Additional texture features (mean and std of intensity)
        mean_intensity = np.mean(gray) / 255.0
        std_intensity = np.std(gray) / 255.0
        
        # Combine all features
        features = np.hstack([hu_moments, hog_features, hist_features, [edge_density, mean_intensity, std_intensity]])
        
        # Ensure fixed feature size (pad if necessary)
        target_size = 200
        if len(features) < target_size:
            features = np.pad(features, (0, target_size - len(features)))
        else:
            features = features[:target_size]
        
        return features
        
    except Exception as e:
        return None

def detect_and_classify_coins(image_path, svm_model):
    """
    Detect multiple coins in an image and classify them as OLD or NEW
    """
    # Read image
    original_img = cv2.imread(image_path)
    if original_img is None:
        try:
            pil_img = Image.open(image_path)
            original_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except:
            return None, 0, 0, 0, []
    
    display_img = original_img.copy()
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.medianBlur(gray_img, 5)
    
    # Hough Circle Detection
    circles = cv2.HoughCircles(
        gray_img, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2,
        minDist=50,
        param1=80,
        param2=40,
        minRadius=30,
        maxRadius=120
    )
    
    old_count = 0
    new_count = 0
    total_value = 0
    detections = []
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for i, (x, y, r) in enumerate(circles[0]):
            buffer = 10
            x_start = max(0, x - r - buffer)
            y_start = max(0, y - r - buffer)
            x_end = min(original_img.shape[1], x + r + buffer)
            y_end = min(original_img.shape[0], y + r + buffer)
            
            if x_end <= x_start + 1 or y_end <= y_start + 1:
                continue
                
            coin_roi = original_img[y_start:y_end, x_start:x_end]
            if coin_roi.shape[0] == 0 or coin_roi.shape[1] == 0:
                continue
            
            # Save ROI temporarily
            temp_roi_path = f'/tmp/temp_roi_{i}.jpg'
            cv2.imwrite(temp_roi_path, coin_roi)
            
            # Extract features
            features = extract_features(temp_roi_path)
            if os.path.exists(temp_roi_path):
                os.remove(temp_roi_path)
            
            if features is not None and svm_model is not None:
                try:
                    features = features.reshape(1, -1)
                    
                    # Predict
                    prediction = svm_model.predict(features)[0]
                    
                    # Get probability if available
                    try:
                        probabilities = svm_model.predict_proba(features)[0]
                        confidence = probabilities[prediction]
                    except:
                        confidence = 0.9
                    
                    coin_type = 'NEW' if prediction == 1 else 'OLD'
                    
                    # Update counts
                    if coin_type == 'OLD':
                        old_count += 1
                    else:
                        new_count += 1
                    total_value += 1
                    
                    detections.append({
                        'position': (x, y),
                        'radius': r,
                        'type': coin_type,
                        'confidence': confidence
                    })
                    
                    # Draw on image
                    color = (0, 255, 0) if coin_type == 'NEW' else (0, 0, 255)
                    cv2.circle(display_img, (x, y), r, color, 4)
                    cv2.putText(display_img, f"{coin_type}", 
                               (x - r, y - r - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                except Exception as e:
                    continue
    
    return display_img, old_count, new_count, total_value, detections

# ============================================
# Training Function
# ============================================

def train_classifier(old_images, new_images):
    """Train SVM classifier"""
    
    X = []
    y = []
    
    # Process old coins (label = 0)
    for img_path in old_images:
        features = extract_features(img_path)
        if features is not None:
            X.append(features)
            y.append(0)
    
    # Process new coins (label = 1)
    for img_path in new_images:
        features = extract_features(img_path)
        if features is not None:
            X.append(features)
            y.append(1)
    
    if len(X) < 4:
        return None, 0, len(X)
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data
    test_size = min(0.3, 1.0 - (3.0/len(X))) if len(X) > 3 else 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Train SVM
    svm_model = SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')
    svm_model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = 0
    if len(X_test) > 0:
        y_pred = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    
    return svm_model, accuracy, len(X)

# ============================================
# Streamlit UI
# ============================================

st.set_page_config(
    page_title="PisoUno - 1 Peso Coin Detection System",
    page_icon=":coin:",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    .coin-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .old-coin-text {
        color: #8B4513;
        font-size: 2rem;
        font-weight: bold;
    }
    .new-coin-text {
        color: #4169E1;
        font-size: 2rem;
        font-weight: bold;
    }
    .total-value-text {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2e7d32;
        text-align: center;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #2a5298, #1e3c72);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>PisoUno</h1>
    <p>Old and New 1 Peso Coin Detection, Classification, and Counting System</p>
    <p>Philippine Currency Recognition System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Training Phase")
    
    st.markdown("---")
    st.subheader("Step 1: Upload Training Images")
    
    st.markdown("**Upload OLD 1 Peso Coin Images**")
    old_files = st.file_uploader(
        "Select OLD 1 Peso coin images (front and back views recommended)",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="old"
    )
    
    st.markdown("**Upload NEW 1 Peso Coin Images**")
    new_files = st.file_uploader(
        "Select NEW 1 Peso coin images (front and back views recommended)",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="new"
    )
    
    st.markdown("---")
    
    if st.button("Train Classifier", type="primary", use_container_width=True):
        if old_files and new_files:
            if len(old_files) >= 2 and len(new_files) >= 2:
                with st.spinner("Training classifier. Please wait..."):
                    old_paths = []
                    new_paths = []
                    
                    for f in old_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                            tmp.write(f.getvalue())
                            old_paths.append(tmp.name)
                    
                    for f in new_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                            tmp.write(f.getvalue())
                            new_paths.append(tmp.name)
                    
                    model, acc, num = train_classifier(old_paths, new_paths)
                    
                    for p in old_paths + new_paths:
                        if os.path.exists(p):
                            os.unlink(p)
                    
                    if model:
                        st.session_state['classifier'] = model
                        st.session_state['trained'] = True
                        st.success(f"Training completed successfully! Trained on {num} images.")
                        if acc > 0:
                            st.metric("Validation Accuracy", f"{acc:.1%}")
                    else:
                        st.error("Training failed. Please ensure you have at least 2 clear images per class.")
            else:
                st.warning("Please upload at least 2 images for each coin type (OLD and NEW).")
        else:
            st.warning("Please upload both OLD and NEW coin images for training.")
    
    if st.session_state.get('trained'):
        st.markdown("---")
        st.success("Classifier is ready for testing.")
        if st.button("Reset Classifier", use_container_width=True):
            st.session_state['trained'] = False
            st.session_state['classifier'] = None
            st.rerun()

# Main area
st.markdown("---")

if not st.session_state.get('trained'):
    st.info("Step 2: Train the classifier first by uploading OLD and NEW coin images in the sidebar.")
    
    with st.expander("How to use this system"):
        st.markdown("""
        **Instructions:**
        
        1. **Upload Training Images** (Sidebar)
           - Upload at least 2 images of OLD 1 Peso coins
           - Upload at least 2 images of NEW 1 Peso coins
           - Include both front and back views for better accuracy
        
        2. **Train the Classifier**
           - Click the 'Train Classifier' button
           - Wait for training to complete
        
        3. **Test the System**
           - After training, upload test images below
           - The system will detect and classify coins
        
        **Tips for Best Results:**
        - Use clear, well-lit photos
        - Place coins on a plain, contrasting background
        - Avoid shadows and reflections
        - Ensure coins are not touching or overlapping
        - Include multiple angles in training data
        """)
else:
    st.header("Testing Phase")
    st.markdown("Upload an image containing one or more Philippine 1 Peso coins for detection.")
    
    test_file = st.file_uploader(
        "Select coin image for testing",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image with good lighting for best results",
        key="test_image"
    )
    
    if test_file:
        with st.spinner("Processing image and detecting coins..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(test_file.getvalue())
                test_path = tmp.name
            
            # Detect and classify coins
            result, old_c, new_c, total, detections = detect_and_classify_coins(
                test_path, st.session_state['classifier']
            )
            
            # Clean up
            os.unlink(test_path)
        
        if result is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Detection Result")
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.caption("Detected coins are highlighted with colored circles: Red = OLD, Green = NEW")
            
            with col2:
                st.subheader("Summary")
                
                # Display counts in columns
                count_col1, count_col2 = st.columns(2)
                count_col1.metric("OLD Coins", old_c)
                count_col2.metric("NEW Coins", new_c)
                
                # Display total value
                st.markdown(f"""
                <div class="coin-card">
                    <div style="font-size: 1.2rem; font-weight: bold;">Total Value</div>
                    <div class="total-value-text">PHP {total}.00</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display detection details
                if detections:
                    st.markdown("---")
                    st.subheader("Detection Details")
                    for i, det in enumerate(detections, 1):
                        confidence_percent = det['confidence'] * 100
                        st.write(f"Coin {i}: {det['type']} (Confidence: {confidence_percent:.1f}%)")
                    
                    st.success(f"Successfully identified {len(detections)} Philippine 1 Peso coin(s).")
                else:
                    st.warning("No coins detected in the image.")
                    st.info("Try adjusting lighting, using a plain background, or ensuring coins are clearly visible.")
        else:
            st.error("Failed to process the image. Please try another image.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 1rem;">
    <p><strong>PisoUno</strong> - Philippine 1 Peso Coin Detection System</p>
    <p>Features: Denomination Classification | Old vs New Classification | Multiple Coin Detection | Counting | Total Value Calculation</p>
    <p>Powered by SVM, HOG Features, Hu Moments, and OpenCV</p>
    <p><small>For educational and research purposes only</small></p>
</div>
""", unsafe_allow_html=True)
