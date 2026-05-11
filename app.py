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


def compute_hog_features(gray_image, cells_per_block=(2, 2), pixels_per_cell=(8, 8), orientations=9):
    """
    Manual HOG feature extraction without scikit-image
    """
    gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180

    h, w = gray_image.shape
    cell_h, cell_w = pixels_per_cell
    block_h, block_w = cells_per_block

    cells_x = w // cell_w
    cells_y = h // cell_h

    hist_bins = orientations
    hist_range = (0, 180)

    cell_histograms = np.zeros((cells_y, cells_x, hist_bins))
    
    for y in range(cells_y):
        for x in range(cells_x):
            cell_mag = magnitude[y*cell_h:(y+1)*cell_h, x*cell_w:(x+1)*cell_w]
            cell_ori = orientation[y*cell_h:(y+1)*cell_h, x*cell_w:(x+1)*cell_w]

            hist, _ = np.histogram(cell_ori.flatten(), bins=hist_bins, range=hist_range, weights=cell_mag.flatten())
            cell_histograms[y, x] = hist

    block_features = []
    for y in range(cells_y - block_h + 1):
        for x in range(cells_x - block_w + 1):
            block = cell_histograms[y:y+block_h, x:x+block_w].flatten()
            block = block / (np.sqrt(np.sum(block**2)) + 1e-6)
            block_features.extend(block)
    
    return np.array(block_features)

def extract_features(image_path, img_size=(64, 64)):
    """
    Extract features from a coin image
    Features: Hu moments, HOG, color histograms, edge density
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            try:
                pil_img = Image.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except:
                return None

        img_resized = cv2.resize(img, img_size)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

        try:
            hog_features = compute_hog_features(gray)
            if len(hog_features) > 100:
                hog_features = hog_features[:100]
        except:
            hog_features = np.zeros(36)

        hist_features = []
        for i in range(3):
            hist = cv2.calcHist([img_resized], [i], None, [8], [0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-7)
            hist_features.extend(hist)

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (img_size[0] * img_size[1])

        mean_intensity = np.mean(gray) / 255.0
        std_intensity = np.std(gray) / 255.0

        features = np.hstack([hu_moments, hog_features, hist_features, [edge_density, mean_intensity, std_intensity]])

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

            temp_roi_path = f'/tmp/temp_roi_{i}.jpg'
            cv2.imwrite(temp_roi_path, coin_roi)

            features = extract_features(temp_roi_path)
            if os.path.exists(temp_roi_path):
                os.remove(temp_roi_path)
            
            if features is not None and svm_model is not None:
                try:
                    features = features.reshape(1, -1)

                    prediction = svm_model.predict(features)[0]

                    try:
                        probabilities = svm_model.predict_proba(features)[0]
                        confidence = probabilities[prediction]
                    except:
                        confidence = 0.9
                    
                    coin_type = 'NEW' if prediction == 1 else 'OLD'

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

                    color = (0, 255, 0) if coin_type == 'NEW' else (0, 0, 255)
                    cv2.circle(display_img, (x, y), r, color, 4)
                    cv2.putText(display_img, f"{coin_type}", 
                               (x - r, y - r - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                except Exception as e:
                    continue
    
    return display_img, old_count, new_count, total_value, detections


def train_classifier(old_images, new_images):
    """Train SVM classifier"""
    
    X = []
    y = []

    for img_path in old_images:
        features = extract_features(img_path)
        if features is not None:
            X.append(features)
            y.append(0)

    for img_path in new_images:
        features = extract_features(img_path)
        if features is not None:
            X.append(features)
            y.append(1)
    
    if len(X) < 4:
        return None, 0, len(X)
    
    X = np.array(X)
    y = np.array(y)

    test_size = min(0.3, 1.0 - (3.0/len(X))) if len(X) > 3 else 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    svm_model = SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')
    svm_model.fit(X_train, y_train)

    accuracy = 0
    if len(X_test) > 0:
        y_pred = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    
    return svm_model, accuracy, len(X)


st.set_page_config(
    page_title="PisoUno - 1 Peso Coin Detection",
    page_icon=":coin:",
    layout="wide"
)

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
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>PisoUno</h1>
    <p>Old and New 1 Peso Coin Detection, Classification, and Counting System</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Training")
    
    st.subheader("Upload Training Images")
    
    old_files = st.file_uploader(
        "OLD 1 Peso Coins",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="old"
    )
    
    new_files = st.file_uploader(
        "NEW 1 Peso Coins",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="new"
    )
    
    if st.button("Train", type="primary"):
        if old_files and new_files:
            if len(old_files) >= 2 and len(new_files) >= 2:
                with st.spinner("Training..."):
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
                        st.success(f"Done! Trained on {num} images")
                        if acc > 0:
                            st.metric("Accuracy", f"{acc:.1%}")
                    else:
                        st.error("Training failed")
            else:
                st.warning("Need at least 2 images per type")
        else:
            st.warning("Upload both OLD and NEW coins")
    
    if st.session_state.get('trained'):
        st.divider()
        st.success("Ready")
        if st.button("Reset"):
            st.session_state['trained'] = False
            st.rerun()

if not st.session_state.get('trained'):
    st.info("Train the classifier first - Upload OLD and NEW coin images in the sidebar")
    
    with st.expander("How to use"):
        st.markdown("""
        1. Upload 2+ images of OLD 1 Peso coins
        2. Upload 2+ images of NEW 1 Peso coins  
        3. Click Train
        4. Upload test images to detect coins
        """)
else:
    st.header("Test")
    
    test_file = st.file_uploader("Upload coin image", type=['jpg', 'jpeg', 'png'])
    
    if test_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(test_file.getvalue())
            test_path = tmp.name
        
        with st.spinner("Detecting..."):
            result, old_c, new_c, total, detections = detect_and_classify_coins(
                test_path, st.session_state['classifier']
            )
        
        os.unlink(test_path)
        
        if result is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            with col2:
                st.subheader("Results")
                
                c1, c2 = st.columns(2)
                c1.metric("OLD", old_c)
                c2.metric("NEW", new_c)
                
                st.markdown(f"""
                <div class="coin-card">
                    <div>TOTAL VALUE</div>
                    <div class="total-value-text">PHP {total}.00</div>
                </div>
                """, unsafe_allow_html=True)
                
                if detections:
                    st.success(f"Found {len(detections)} coin(s)")
                else:
                    st.warning("No coins detected")
        else:
            st.error("Failed to process")

st.divider()
st.caption("PisoUno - Philippine 1 Peso Coin Detection System")
