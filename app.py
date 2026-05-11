import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from PIL import Image
import os
import tempfile
import hashlib
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Try to import HEIC support
try:
    import pillow_heif
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False
    print("HEIC support not available. Install pillow_heif for HEIC file support.")

def convert_heic_to_png(heic_path):
    """Convert HEIC image to PNG format"""
    if not HEIC_SUPPORT:
        return None
    try:
        pillow_heif.register_heif_opener()
        heif_file = Image.open(heic_path)
        png_path = heic_path.replace('.HEIC', '.png').replace('.heic', '.png')
        heif_file.save(png_path, format="png")
        return png_path
    except Exception as e:
        st.error(f"Error converting HEIC file: {e}")
        return None

def extract_features(image_path, img_size=(64, 64), cache_dir=None):
    """
    Extract features from a coin image
    Features: Hu moments, HOG, color histograms, edge density
    """
    image_hash = hashlib.md5(image_path.encode('utf-8')).hexdigest()
    
    if cache_dir:
        cache_file = os.path.join(cache_dir, f'{image_hash}.npy')
        if os.path.exists(cache_file):
            try:
                return np.load(cache_file)
            except:
                if os.path.exists(cache_file):
                    os.remove(cache_file)
    
    # Handle HEIC files
    processed_image_path = image_path
    if image_path.lower().endswith(('.heic')):
        processed_image_path = convert_heic_to_png(image_path)
        if processed_image_path is None:
            return None
    
    try:
        img = cv2.imread(processed_image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {processed_image_path}")
        
        img_resized = cv2.resize(img, img_size)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # Hu Moments (7 features)
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        # HOG Features
        hog_features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                          orientations=9, block_norm='L2-Hys', visualize=False)
        
        # Color Histograms (3 channels × 8 bins = 24 features)
        hist_features = []
        for i in range(3):
            hist = cv2.calcHist([img_resized], [i], None, [8], [0, 256])
            hist_features.extend(hist.flatten())
        
        # Edge Density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (img_size[0] * img_size[1])
        
        # Combine all features
        features = np.hstack([hu_moments, hog_features, hist_features, edge_density])
        
        # Cache features
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_file, features)
        
        # Clean up temporary HEIC conversion
        if image_path.lower().endswith(('.heic')) and os.path.exists(processed_image_path):
            os.remove(processed_image_path)
        
        return features
        
    except Exception as e:
        st.error(f"Error processing image {image_path}: {e}")
        if image_path.lower().endswith(('.heic')) and os.path.exists(processed_image_path):
            os.remove(processed_image_path)
        return None

def load_and_extract_features_from_upload(uploaded_files, coin_type, cache_dir):
    """Load uploaded images and extract features"""
    features_list = []
    success_count = 0
    
    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        # Extract features
        features = extract_features(temp_path, cache_dir=cache_dir)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        if features is not None:
            features_list.append(features)
            success_count += 1
    
    return np.array(features_list) if features_list else None

def detect_and_classify_coins(image_path, svm_model, feature_extractor_func):
    """
    Detect multiple coins in an image and classify them as OLD or NEW
    """
    # Handle HEIC files
    processed_image_path = image_path
    if image_path.lower().endswith(('.heic')):
        processed_image_path = convert_heic_to_png(image_path)
        if processed_image_path is None:
            return None, 0, 0, 0
    
    original_img = cv2.imread(processed_image_path)
    if original_img is None:
        if image_path.lower().endswith(('.heic')) and os.path.exists(processed_image_path):
            os.remove(processed_image_path)
        return None, 0, 0, 0
    
    display_img = original_img.copy()
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.medianBlur(gray_img, 5)
    
    # Hough Circle Detection
    circles = cv2.HoughCircles(
        gray_img, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=70,
        param1=100, 
        param2=50, 
        minRadius=50, 
        maxRadius=100
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
            temp_roi_path = f'./temp_roi_{i}.jpg'
            cv2.imwrite(temp_roi_path, coin_roi)
            
            # Extract features
            features = feature_extractor_func(temp_roi_path)
            os.remove(temp_roi_path)
            
            if features is not None:
                if features.shape[0] != svm_model.n_features_in_:
                    continue
                
                # Predict
                prediction = svm_model.predict(features.reshape(1, -1))[0]
                probabilities = svm_model.predict_proba(features.reshape(1, -1))[0]
                
                coin_type = 'NEW' if prediction == 1 else 'OLD'
                confidence = probabilities[prediction]
                
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
                cv2.putText(display_img, f"{coin_type} ({confidence:.2f})", 
                           (x - r, y - r - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Clean up
    if image_path.lower().endswith(('.heic')) and os.path.exists(processed_image_path):
        os.remove(processed_image_path)
    
    return display_img, old_count, new_count, total_value, detections


def train_classifier(old_images, new_images, use_smote=True):
    """Train SVM classifier with optional SMOTE balancing"""
    
    with st.spinner("Extracting features from training images..."):
        # Extract features
        X = []
        y = []
        
        # Process old coins
        progress_bar = st.progress(0)
        for i, img in enumerate(old_images):
            features = extract_features(img, cache_dir=None)
            if features is not None:
                X.append(features)
                y.append(0)
            progress_bar.progress((i + 1) / (len(old_images) + len(new_images)))
        
        # Process new coins
        for i, img in enumerate(new_images):
            features = extract_features(img, cache_dir=None)
            if features is not None:
                X.append(features)
                y.append(1)
            progress_bar.progress((len(old_images) + i + 1) / (len(old_images) + len(new_images)))
        
        if len(X) == 0:
            st.error("No valid images found for training!")
            return None, None, 0
        
        X = np.array(X)
        y = np.array(y)
        
        st.info(f"Extracted {len(X)} feature vectors (Dimension: {X.shape[1]})")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Apply SMOTE if requested
        if use_smote and len(np.unique(y_train)) == 2:
            try:
                sm = SMOTE(random_state=42)
                X_train, y_train = sm.fit_resample(X_train, y_train)
                st.info(f"Applied SMOTE. Training set size: {len(X_train)}")
            except Exception as e:
                st.warning(f"SMOTE failed: {e}. Continuing without SMOTE.")
    
    # Train SVM
    with st.spinner("Training SVM classifier..."):
        svm_model = SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')
        svm_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return svm_model, accuracy, len(X)


st.set_page_config(
    page_title="PisoUno - 1 Peso Coin Detection System",
    page_icon="🪙",
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
        font-size: 1.5rem;
        font-weight: bold;
    }
    .new-coin-text {
        color: #4169E1;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .total-value-text {
        font-size: 2rem;
        font-weight: bold;
        color: #2e7d32;
        text-align: center;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
        border: none;
        padding: 0.5rem;
        font-size: 1rem;
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
    <p><small>Powered by SVM + HOG + Hu Moments | Supports JPEG, PNG, HEIC</small></p>
</div>
""", unsafe_allow_html=True)

# Sidebar for training
with st.sidebar:
    st.header("Training Phase")
    st.markdown("Upload sample images to train the classifier.")
    
    # Training section
    st.subheader("Step 1: Upload Training Images")
    
    old_training_files = st.file_uploader(
        "OLD 1 Peso Coins (Front/Back views)",
        type=['jpg', 'jpeg', 'png', 'heic', 'HEIC'],
        accept_multiple_files=True,
        key="old_train"
    )
    
    new_training_files = st.file_uploader(
        "NEW 1 Peso Coins (Front/Back views)",
        type=['jpg', 'jpeg', 'png', 'heic', 'HEIC'],
        accept_multiple_files=True,
        key="new_train"
    )
    
    use_smote = st.checkbox("Use SMOTE for class balancing", value=True)
    
    if st.button("Train Classifier", type="primary", use_container_width=True):
        if old_training_files and new_training_files:
            with st.spinner("Processing training images..."):
                # Save uploaded files temporarily
                old_paths = []
                new_paths = []
                
                # Save old images
                for file in old_training_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                        tmp.write(file.getvalue())
                        old_paths.append(tmp.name)
                
                # Save new images
                for file in new_training_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                        tmp.write(file.getvalue())
                        new_paths.append(tmp.name)
                
                # Train classifier
                model, accuracy, num_samples = train_classifier(old_paths, new_paths, use_smote)
                
                # Clean up temp files
                for path in old_paths + new_paths:
                    if os.path.exists(path):
                        os.unlink(path)
                
                if model is not None:
                    st.session_state['classifier'] = model
                    st.session_state['trained'] = True
                    st.session_state['training_samples'] = num_samples
                    
                    st.success(f"Training Complete!")
                    st.metric("Test Accuracy", f"{accuracy:.1%}")
                    st.info(f"Trained on {num_samples} images")
                else:
                    st.error("Training failed. Please check your images.")
        else:
            st.warning("Please upload both OLD and NEW coin images for training.")
    
    # Session status
    if st.session_state.get('trained', False):
        st.divider()
        st.success(f"Classifier Ready! ({st.session_state.get('training_samples', 0)} training samples)")
        
        if st.button("Reset Classifier", use_container_width=True):
            st.session_state['trained'] = False
            st.session_state['classifier'] = None
            st.rerun()

# Main content
st.header("Detection Phase")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("Upload an image containing one or more Philippine 1 Peso coins.")

# File upload for testing
test_file = st.file_uploader(
    "Upload coin image for detection",
    type=['jpg', 'jpeg', 'png', 'heic', 'HEIC'],
    help="Supports multiple coins in one image. Best results with good lighting and plain background.",
    key="test_image"
)

# Check if classifier is trained
if not st.session_state.get('trained', False):
    st.warning("**Classifier not trained yet!**")
    st.info("Please go to the sidebar to upload training images and train the classifier first.")
    
    # Show instructions
    with st.expander("How to use PisoUno"):
        st.markdown("""
        ### Step-by-Step Guide:
        
        1. **Train the System** (Sidebar)
           - Upload 5-10 images of OLD 1 Peso coins (both front and back views recommended)
           - Upload 5-10 images of NEW 1 Peso coins (both front and back views recommended)
           - Click "Train Classifier"
           - Wait for training to complete
        
        2. **Test the System** (Main Area)
           - Upload an image with 1 Peso coins
           - View detection results with annotations
           - See counts and total value
        
        ### Tips for Best Results:
        - Use clear, well-lit photos
        - Place coins on a plain, contrasting background
        - Avoid shadows and reflections
        - Ensure coins don't touch or overlap
        - Include multiple angles in training (front, back, slightly tilted)
        """)

else:
    # Process test image
    if test_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(test_file.name)[1]) as tmp_file:
            tmp_file.write(test_file.getvalue())
            test_path = tmp_file.name
        
        # Detect coins
        with st.spinner("Detecting and classifying coins..."):
            result_img, old_count, new_count, total_value, detections = detect_and_classify_coins(
                test_path,
                st.session_state['classifier'],
                extract_features
            )
        
        # Clean up temp file
        os.unlink(test_path)
        
        if result_img is not None:
            # Display results
            col_result, col_stats = st.columns([2, 1])
            
            with col_result:
                st.subheader("Detection Result")
                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, caption="Annotated Detection", use_container_width=True)
            
            with col_stats:
                st.subheader("Summary")
                
                # Display counts with custom styling
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""
                    <div class="coin-card">
                        <div style="font-size: 1rem;">OLD Coins</div>
                        <div class="old-coin-text">{old_count}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_b:
                    st.markdown(f"""
                    <div class="coin-card">
                        <div style="font-size: 1rem;">NEW Coins</div>
                        <div class="new-coin-text">{new_count}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="coin-card" style="margin-top: 1rem;">
                    <div style="font-size: 1rem;">TOTAL VALUE</div>
                    <div class="total-value-text">₱{total_value}.00</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed detections
                if detections:
                    st.divider()
                    st.write("**Detailed Results:**")
                    for i, det in enumerate(detections, 1):
                        emoji = "OLD" if det['type'] == 'OLD' else "NEW"
                        st.write(f"{emoji} Coin {i}: **{det['type']}** (confidence: {det['confidence']:.1%})")
                    
                    # Total coins
                    total_coins = old_count + new_count
                    st.info(f"Total coins detected: {total_coins}")
            
            # Success message
            if old_count > 0 or new_count > 0:
                st.success(f"Successfully identified {old_count + new_count} Philippine 1 Peso coin(s)!")
            else:
                st.warning("No coins detected. Try adjusting lighting or coin placement.")
                
                with st.expander("Detection Tips"):
                    st.markdown("""
                    - Ensure good lighting (avoid shadows)
                    - Use a plain, contrasting background
                    - Make sure coins are clearly separated
                    - Avoid blurry or out-of-focus images
                    - Try different angles
                    """)
        else:
            st.error("Failed to process the image. Please try another image.")
    else:
        st.info("Please upload an image to start detection.")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; padding: 1rem;">
    <p>PisoUno - Philippine 1 Peso Coin Detection System</p>
    <p>Features: Denomination Classification | Old/New Classification | Multiple Coin Detection | Counting | Total Value</p>
    <p><small>Powered by SVM, HOG Features, Hu Moments, and OpenCV</small></p>
</div>
""", unsafe_allow_html=True)