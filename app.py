import streamlit as st
import sys
import importlib

missing_packages = []

try:
    import cv2
except ImportError:
    missing_packages.append("opencv-python")

try:
    import numpy as np
except ImportError:
    missing_packages.append("numpy")

try:
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except ImportError:
    missing_packages.append("scikit-learn")

try:
    from skimage.feature import hog
except ImportError:
    missing_packages.append("scikit-image")

try:
    from PIL import Image
except ImportError:
    missing_packages.append("Pillow")

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    missing_packages.append("imbalanced-learn")

if missing_packages:
    st.error(f"Missing required packages: {', '.join(missing_packages)}")
    st.info("Please install missing packages using: pip install " + " ".join(missing_packages))
    st.stop()

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
import warnings
warnings.filterwarnings('ignore')

HEIC_SUPPORT = False
try:
    import pillow_heif
    HEIC_SUPPORT = True
except ImportError:
    pass


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

    processed_image_path = image_path
    if image_path.lower().endswith(('.heic')):
        processed_image_path = convert_heic_to_png(image_path)
        if processed_image_path is None:
            return None
    
    try:
        img = cv2.imread(processed_image_path)
        if img is None:
            try:
                pil_img = Image.open(processed_image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except:
                raise FileNotFoundError(f"Image not found at {processed_image_path}")
        
        img_resized = cv2.resize(img, img_size)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        try:
            hog_features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                              orientations=9, block_norm='L2-Hys', visualize=False)
        except:
            hog_features = np.zeros(36)  

        hist_features = []
        for i in range(3):
            hist = cv2.calcHist([img_resized], [i], None, [8], [0, 256])
            hist_features.extend(hist.flatten())

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (img_size[0] * img_size[1])

        features = np.hstack([hu_moments, hog_features, hist_features, edge_density])

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_file, features)

        if image_path.lower().endswith(('.heic')) and os.path.exists(processed_image_path):
            os.remove(processed_image_path)
        
        return features
        
    except Exception as e:
        if image_path.lower().endswith(('.heic')) and os.path.exists(processed_image_path):
            os.remove(processed_image_path)
        return None

def detect_and_classify_coins(image_path, svm_model, feature_extractor_func):
    """
    Detect multiple coins in an image and classify them as OLD or NEW
    """

    processed_image_path = image_path
    if image_path.lower().endswith(('.heic')):
        processed_image_path = convert_heic_to_png(image_path)
        if processed_image_path is None:
            return None, 0, 0, 0, []
    
    original_img = cv2.imread(processed_image_path)
    if original_img is None:
        try:
            pil_img = Image.open(processed_image_path)
            original_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except:
            if image_path.lower().endswith(('.heic')) and os.path.exists(processed_image_path):
                os.remove(processed_image_path)
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

            temp_roi_path = f'./temp_roi_{i}.jpg'
            cv2.imwrite(temp_roi_path, coin_roi)

            features = feature_extractor_func(temp_roi_path)
            if os.path.exists(temp_roi_path):
                os.remove(temp_roi_path)
            
            if features is not None and svm_model is not None:
                try:
                    if features.shape[0] != svm_model.n_features_in_:
                        continue

                    prediction = svm_model.predict(features.reshape(1, -1))[0]
                    probabilities = svm_model.predict_proba(features.reshape(1, -1))[0]
                    
                    coin_type = 'NEW' if prediction == 1 else 'OLD'
                    confidence = probabilities[prediction] if prediction < len(probabilities) else 0.5

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
                    cv2.putText(display_img, f"{coin_type} ({confidence:.2f})", 
                               (x - r, y - r - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                except Exception as e:
                    continue

    if image_path.lower().endswith(('.heic')) and os.path.exists(processed_image_path):
        os.remove(processed_image_path)
    
    return display_img, old_count, new_count, total_value, detections



def train_classifier(old_images, new_images, use_smote=True):
    """Train SVM classifier with optional SMOTE balancing"""
    
    X = []
    y = []

    for i, img in enumerate(old_images):
        features = extract_features(img, cache_dir=None)
        if features is not None:
            X.append(features)
            y.append(0)

    for i, img in enumerate(new_images):
        features = extract_features(img, cache_dir=None)
        if features is not None:
            X.append(features)
            y.append(1)
    
    if len(X) == 0:
        return None, 0, 0
    
    X = np.array(X)
    y = np.array(y)

    if len(X) < 5:
        return None, 0, len(X)
    
    test_size = min(0.2, 1.0 - (3.0/len(X))) if len(X) > 3 else 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if use_smote and len(np.unique(y_train)) == 2 and len(y_train) > 5:
        try:
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
        except Exception:
            pass

    svm_model = SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')
    svm_model.fit(X_train, y_train)

    accuracy = 0
    if len(X_test) > 0:
        y_pred = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    
    return svm_model, accuracy, len(X)


st.set_page_config(
    page_title="PisoUno - 1 Peso Coin Detection System",
    page_icon="🪙",
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

st.markdown("""
<div class="main-header">
    <h1>PisoUno</h1>
    <p>Old and New 1 Peso Coin Detection, Classification, and Counting System</p>
    <p><small>Powered by SVM + HOG + Hu Moments</small></p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Training Phase")
    st.markdown("Upload sample images to train the classifier.")
    
    st.subheader("Step 1: Upload Training Images")
    
    old_training_files = st.file_uploader(
        "OLD 1 Peso Coins (Front/Back views)",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="old_train"
    )
    
    new_training_files = st.file_uploader(
        "NEW 1 Peso Coins (Front/Back views)",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="new_train"
    )
    
    use_smote = st.checkbox("Use SMOTE for class balancing", value=True)
    
    if st.button("Train Classifier", type="primary", use_container_width=True):
        if old_training_files and new_training_files:
            with st.spinner("Processing training images..."):
                old_paths = []
                new_paths = []

                for file in old_training_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        tmp.write(file.getvalue())
                        old_paths.append(tmp.name)

                for file in new_training_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        tmp.write(file.getvalue())
                        new_paths.append(tmp.name)

                model, accuracy, num_samples = train_classifier(old_paths, new_paths, use_smote)

                for path in old_paths + new_paths:
                    if os.path.exists(path):
                        os.unlink(path)
                
                if model is not None:
                    st.session_state['classifier'] = model
                    st.session_state['trained'] = True
                    st.session_state['training_samples'] = num_samples
                    
                    st.success(f"Training Complete!")
                    if accuracy > 0:
                        st.metric("Test Accuracy", f"{accuracy:.1%}")
                    st.info(f"Trained on {num_samples} images")
                else:
                    st.error("Training failed. Please check your images (need at least 2 per class).")
        else:
            st.warning("Please upload both OLD and NEW coin images for training.")

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

test_file = st.file_uploader(
    "Upload coin image for detection",
    type=['jpg', 'jpeg', 'png'],
    help="For best results: good lighting, plain background, coins not touching.",
    key="test_image"
)

if not st.session_state.get('trained', False):
    st.warning("**Classifier not trained yet!**")
    st.info("Please go to the sidebar to upload training images and train the classifier first.")
    
    with st.expander("How to use PisoUno"):
        st.markdown("""
        ### Step-by-Step Guide:
        
        1. **Train the System** (Sidebar)
           - Upload 5-10 images of OLD 1 Peso coins
           - Upload 5-10 images of NEW 1 Peso coins
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
        """)

else:
    if test_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(test_file.getvalue())
            test_path = tmp_file.name
        
        with st.spinner("Detecting and classifying coins..."):
            result_img, old_count, new_count, total_value, detections = detect_and_classify_coins(
                test_path,
                st.session_state['classifier'],
                extract_features
            )

        os.unlink(test_path)
        
        if result_img is not None:
            col_result, col_stats = st.columns([2, 1])
            
            with col_result:
                st.subheader("Detection Result")
                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, caption="Annotated Detection", use_container_width=True)
            
            with col_stats:
                st.subheader("Summary")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""
                    <div class="coin-card">
                        <div>OLD Coins</div>
                        <div class="old-coin-text">{old_count}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_b:
                    st.markdown(f"""
                    <div class="coin-card">
                        <div>NEW Coins</div>
                        <div class="new-coin-text">{new_count}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="coin-card" style="margin-top: 1rem;">
                    <div>TOTAL VALUE</div>
                    <div class="total-value-text">₱{total_value}.00</div>
                </div>
                """, unsafe_allow_html=True)
                
                if detections:
                    st.divider()
                    st.write("**Detailed Results:**")
                    for i, det in enumerate(detections, 1):
                        emoji = "OLD" if det['type'] == 'OLD' else "NEW"
                        st.write(f"{emoji} Coin {i}: **{det['type']}**")
                    
                    st.info(f"Total coins detected: {len(detections)}")
            
            if old_count > 0 or new_count > 0:
                st.success(f"Successfully identified {old_count + new_count} Philippine 1 Peso coin(s)!")
            else:
                st.warning("No coins detected. Try adjusting lighting or coin placement.")
        else:
            st.error("Failed to process the image. Please try another image.")
    else:
        st.info("Please upload an image to start detection.")

st.divider()
st.markdown("""
<div style="text-align: center; color: gray; padding: 1rem;">
    <p>PisoUno - Philippine 1 Peso Coin Detection System</p>
    <p>Features: Classification | Multiple Coin Detection | Counting | Total Value</p>
</div>
""", unsafe_allow_html=True)
