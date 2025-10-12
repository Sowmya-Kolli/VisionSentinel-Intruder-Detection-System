import streamlit as st
import cv2
import numpy as np
import os
import json
import hashlib
import shutil
from datetime import datetime
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import uuid
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---- MODERN UI STYLES (optional, you can add your CSS here) ----

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700;900&display=swap');
body, .stApp {
    background: linear-gradient(120deg, #181c24 0%, #232a36 100%);
    font-family: 'Montserrat', 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    color: #e6eaf3;
}

/* Keyframes for fade-in and slide-up */
@keyframes fadeInUp {
    0% {
        opacity: 0;
        transform: translateY(30px);
    }
    100% {
        opacity: 1;
        transform: translateY(0);
    }
}
@keyframes fadeIn {
    0% { opacity: 0; }
    100% { opacity: 1; }
}

.stApp > header, .block-container, .stFileUploader, .stCameraInput, .stExpander, .stAlert, .stDownloadButton, .stButton, .stImage, .stColumns {
    background: rgba(32, 38, 50, 0.82) !important;
    backdrop-filter: blur(12px);
    border-radius: 22px;
    box-shadow: 0 8px 32px 0 rgba(20, 24, 31, 0.25), 0 2px 8px rgba(0,0,0,0.10);
    border: 2.5px solid #232a36;
    transition: box-shadow 0.3s, border 0.3s;
    animation: fadeInUp 0.9s cubic-bezier(.39,.575,.565,1) both;
}

[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #232a36 70%, #00b894 100%);
    color: #e6eaf3;
    border-top-right-radius: 22px;
    border-bottom-right-radius: 22px;
    box-shadow: 2px 0 18px 0 rgba(20, 24, 31, 0.18);
    padding-top: 1.5em;
    animation: fadeIn 1.2s ease-in;
}

h1, h2, h3, .stTitle, .stHeader, .stSubheader {
    color: #00b894 !important;
    font-family: 'Montserrat', 'Segoe UI', sans-serif;
    font-weight: 900;
    letter-spacing: 1.2px;
    text-shadow: 0 2px 12px #00b89444, 0 1px 2px #181c2433;
    animation: fadeInUp 0.9s cubic-bezier(.39,.575,.565,1) both;
}

.stCaption, .stMarkdown p {
    color: #a6b0c3 !important;
    font-size: 1.08rem;
    margin-bottom: 1rem;
    font-weight: 500;
    animation: fadeIn 1.2s;
}

.stAlert {
    border-radius: 16px;
    box-shadow: 0 2px 16px #00b89433;
    padding: 1.2em 1.5em;
    animation: fadeInUp 1s cubic-bezier(.39,.575,.565,1) both;
}

.stButton>button, .stDownloadButton>button {
    background: linear-gradient(90deg, #00b894 60%, #00cec9 100%);
    color: #181c24;
    border: none;
    border-radius: 32px;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 0.7em 2.2em;
    margin: 0.3em 0.2em;
    box-shadow: 0 2px 16px #00b89444;
    transition: background 0.3s, box-shadow 0.3s, transform 0.2s;
    animation: fadeInUp 1.1s;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background: linear-gradient(90deg, #00cec9 60%, #00b894 100%);
    color: #232a36;
    transform: scale(1.04);
    box-shadow: 0 4px 24px #00b89455;
}
.stFileUploader>div>div>button {
    background: #00b894 !important;
    color: #232a36 !important;
    border-radius: 18px !important;
    animation: fadeInUp 1.2s;
}
.stImage>img, .stImage>div>img {
    border-radius: 18px;
    box-shadow: 0 4px 24px #00b89433;
    transition: box-shadow 0.3s, transform 0.2s;
    animation: fadeInUp 1.2s;
}
.stImage>img:hover, .stImage>div>img:hover {
    box-shadow: 0 8px 32px #00cec922;
    transform: scale(1.01);
}
hr {
    border: none;
    border-top: 2px solid #00b89444;
    margin: 2em 0;
}
</style>
""", unsafe_allow_html=True)


# ---- VisionSentinel Branding ----
st.markdown("""
<div style='text-align:center; margin-bottom: 1.2em;'>
    <h1 style='font-family:Montserrat,Segoe UI,sans-serif; font-size:3.2rem; font-weight:900; color:#00b894; letter-spacing:2px; margin-bottom:0.2em; text-shadow:0 4px 16px #181c2433;'>VisionSentinel</h1>
    <div style='font-size:1.25rem; color:#a6b0c3; letter-spacing:1.1px; font-style:italic; margin-bottom:0.2em;'>
        “An intelligent security system guarding every frame.”
    </div>
</div>
""", unsafe_allow_html=True)


# ---- Sidebar Branding ----
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; margin-bottom:1em;'>
        <img src="https://img.icons8.com/fluency/96/security-checked.png" width="64" style="margin-bottom:0.5em;"/>
        <h2 style='font-family:Montserrat,sans-serif; color:#00b894; font-weight:800; letter-spacing:1.2px; margin-bottom:0.3em;'>VisionSentinel</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")


# --- Directories and Constants ---
GALLERY_DIR = "gallery"
NORMAL_DIR = "normal_faces"
INTRUDER_DIR = "intruder_faces"
INTRUDER_MODEL = "intruder_lbph.yml"   
FACE_SIZE = (200, 200)
USERS_FILE = "users.json"
SNAPSHOT_DIR = "intruder_snapshots"
NOT_INTRUDER_SNAPSHOT_DIR = "not_intruder_snapshots"
AGE_MODEL = "age_net.caffemodel"
AGE_PROTO = "age_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']


for d in [GALLERY_DIR, NORMAL_DIR, INTRUDER_DIR]:
    os.makedirs(d, exist_ok=True)

# --- User Management Helpers ---
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    else:
        # Default admin user (hashed password)
        return {"admin": hashlib.sha256("admin123".encode()).hexdigest()}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- Persistent Session State for Users ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "users" not in st.session_state:
    st.session_state.users = load_users()

# --- Helper: Save Face Image from Array ---
def save_face_image_from_array(face_img_array, is_intruder):
    prefix = "intruder" if is_intruder else "normal"
    filename = f"{prefix}{datetime.now().strftime('%Y%m%d%H%M%S_%f')}.png"
    path = os.path.join(INTRUDER_DIR if is_intruder else NORMAL_DIR, filename)
    cv2.imwrite(path, face_img_array)
    return path

# --- Helper: Crop, Detect, and Save Face from PIL Image ---
def save_snapshot(image, is_intruder):
    # Ensure directories exist
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(NOT_INTRUDER_SNAPSHOT_DIR, exist_ok=True)
    # Create unique filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"snapshot_{timestamp}_{uuid.uuid4().hex[:8]}.png"
    # Save to the correct folder
    if is_intruder:
        path = os.path.join(SNAPSHOT_DIR, filename)
    else:
        path = os.path.join(NOT_INTRUDER_SNAPSHOT_DIR, filename)
    cv2.imwrite(path, image)
    return path

def detect_and_crop_face(image_pil):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    img_cv = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda rect: rect[2]*rect[3])
    face_img = gray[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, FACE_SIZE)
    return Image.fromarray(face_img)

def save_face_image(image_pil, save_dir, prefix):
    face_img = detect_and_crop_face(image_pil)
    if face_img is None:
        return None
    filename = f"{prefix}{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    path = os.path.join(save_dir, filename)
    face_img.save(path)
    return path

def save_intruder_image(image_pil):
    """
    Detects and crops the face from a PIL image and saves it to the intruder directory.
    Returns the saved file path or None if no face is detected.
    """
    return save_face_image(image_pil, INTRUDER_DIR, "intruder")


def get_images_and_labels(directory):
    images, labels = [], []
    for fname in os.listdir(directory):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(directory, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, FACE_SIZE)
                images.append(img)
                labels.append(0)
    return images, labels

def train_intruder_model():
    images, labels = get_images_and_labels(INTRUDER_DIR)
    if len(images) == 0:
        return False
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    recognizer.save(INTRUDER_MODEL)
    return True

def load_intruder_model():
    if not os.path.exists(INTRUDER_MODEL):
        return None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(INTRUDER_MODEL)
    return recognizer

def get_snapshot_paths(is_intruder=True):
    dir_path = SNAPSHOT_DIR if is_intruder else NOT_INTRUDER_SNAPSHOT_DIR
    return [os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def clear_snapshots(is_intruder=True):
    dir_path = SNAPSHOT_DIR if is_intruder else NOT_INTRUDER_SNAPSHOT_DIR
    shutil.rmtree(dir_path, ignore_errors=True)
    os.makedirs(dir_path, exist_ok=True)

def load_age_gender_models():
    age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
    return age_net, gender_net

def predict_age_gender(face_img, age_net, gender_net):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.426, 87.768, 114.895), swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_LIST[age_preds[0].argmax()]
    return gender, age


# --- Login/Signup Page ---
def login_page():
    st.title("🔒 Login ")
    st.markdown("Don't have an account? [Sign up below](#signup)")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        hashed_pw = hash_password(password)
        if username in st.session_state.users and st.session_state.users[username] == hashed_pw:
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.markdown("---")
    st.header("Sign up", anchor="signup")
    new_user = st.text_input("New Username")
    new_pass = st.text_input("New Password", type="password")
    if st.button("Sign Up"):
        if new_user and new_pass:
            if new_user in st.session_state.users:
                st.warning("Username already exists.")
            else:
                st.session_state.users[new_user] = hash_password(new_pass)
                save_users(st.session_state.users)
                st.success("Account created! Please login above.")
        else:
            st.warning("Please fill in both fields.")

if not st.session_state.logged_in:
    login_page()
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("Settings")
page = st.sidebar.radio("Go to", ["Camera", "Gallery", "Intruder","Intruder Age and Gender"], index=0)
st.sidebar.markdown("---")

# --- Camera Page ---
if page == "Camera":
    st.title("📷 Camera")
    st.caption("Stream your webcam, adjust settings, and capture images to your gallery.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access camera. Please check your webcam.")
    else:
        with st.expander("Adjustments", expanded=True):
            brightness = st.slider("Brightness", 0.5, 3.0, 1.0, step=0.1)
            aspect_options = {
                "16:9": 16/9, "4:3": 4/3, "1:1": 1.0, "3:4": 3/4, "9:16": 9/16
            }
            aspect_label = st.selectbox("Aspect Ratio", list(aspect_options.keys()), index=0)
            aspect_ratio = aspect_options[aspect_label]
            grayscale = st.checkbox("Grayscale Mode", value=False)
            st.markdown("---")
            capture = st.button("Capture Adjusted Image")

        FRAME_WINDOW = st.empty()
        info_placeholder = st.empty()

        def crop_to_aspect(frame, aspect_ratio):
            h, w = frame.shape[:2]
            target_w = w
            target_h = int(w / aspect_ratio)
            if target_h > h:
                target_h = h
                target_w = int(h * aspect_ratio)
            x1 = (w - target_w) // 2
            y1 = (h - target_h) // 2
            return frame[y1:y1+target_h, x1:x1+target_w]

        def apply_adjustments(frame, brightness, aspect_ratio, grayscale):
            frame = crop_to_aspect(frame, aspect_ratio)
            frame = cv2.convertScaleAbs(frame, alpha=brightness, beta=0)
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            return frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                info_placeholder.warning("Failed to grab frame")
                break
            adjusted = apply_adjustments(frame, brightness, aspect_ratio, grayscale)
            FRAME_WINDOW.image(adjusted, channels="BGR", use_container_width=True)

            if capture:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(GALLERY_DIR, f"capture_{timestamp}.jpg")
                cv2.imwrite(filename, adjusted)
                info_placeholder.success(f"Image saved to gallery as {filename}")
                break

        cap.release()
    st.info("Tip: Use the Gallery to view and manage your captured images.")

# --- Gallery Page ---
elif page == "Gallery":
    st.title("🖼 Gallery")
    st.caption("Browse, preview, and download your captured, not intruder, or intruder images.")

    # Directory mapping for each tab

    gallery_tab = st.tabs(["Captured", "Not Intruder", "Intruder"])
    tab_dirs = [
        GALLERY_DIR,                      # Captured images (manual/camera)
        NOT_INTRUDER_SNAPSHOT_DIR,        # Not intruder detection snapshots
        SNAPSHOT_DIR                      # Intruder detection snapshots
    ]
    tab_msgs = [
        "No captured images yet. Capture some from the Camera page!",
        "No not intruder images yet. Detect some from the Intruder page!",
        "No intruder images yet. Detect some from the Intruder page!"
    ]

    for i, tab in enumerate(gallery_tab):
        with tab:
            img_dir = tab_dirs[i]
            empty_msg = tab_msgs[i]
            if not os.path.exists(img_dir):
                os.makedirs(img_dir, exist_ok=True)
            images = [img for img in os.listdir(img_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images:
                st.info(empty_msg)
            else:
                cols = st.columns(3)
                for idx, img_file in enumerate(sorted(images, reverse=True)):
                    img_path = os.path.join(img_dir, img_file)
                    img = Image.open(img_path)
                    with cols[idx % 3]:
                        st.image(img, caption=img_file, use_container_width=True)
                        with open(img_path, "rb") as f:
                            st.download_button("Download", f, file_name=img_file, mime="image/jpeg", key=f"dl_{i}_{img_file}")
                        if st.button("Delete", key=f"del_{i}_{img_file}"):
                            os.remove(img_path)
                            st.rerun()


# --- Intruder Page ---
elif page == "Intruder":
    st.title("🕵 Intruder Management")
    intruder_nav = st.radio(
        "Intruder Section",
        ["Train Intruder", "Detect Intruder", "Gallery"],
        horizontal=True
    )

    # --- Train Intruder ---
    if intruder_nav == "Train Intruder":

        st.subheader("Register Intruder Images (Multiple Inputs)")

        col1, col2 = st.columns(2)

        with col1:
            uploaded_files = st.file_uploader(
                "Upload Intruder Face Images",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                help="Upload multiple images of the intruder for better recognition."
            )
            if uploaded_files:
                for uploaded in uploaded_files:
                    image = Image.open(uploaded)
                    path = save_intruder_image(image)
                    if path is not None:
                        st.success(f"Face detected and saved: {os.path.basename(path)}")
                    else:
                        st.warning(f"No face detected in {uploaded.name}. Please try another image.")

        with col2:
        
            st.write(
                "Take a photo, it will be saved automatically if a face is detected. "
                "Use the 'X' to clear and take another. Repeat as many times as you want."
            )
            camera_img = st.camera_input("Capture Intruder Face", key="camera_img")
            if camera_img is not None:
                image = Image.open(camera_img)
                path = save_intruder_image(image)
                if path is not None:
                    st.success(f"Face detected and saved: {os.path.basename(path)}")
                    registered = True
                else:
                    st.warning("No face detected in camera capture. Please try again.")

        # After registering images, retrain the model
        if st.button("Train Model"):
            if train_intruder_model():
                st.success("Intruder model trained/updated successfully.")
            else:
                st.warning("No intruder faces found to train the model.")

    # --- Detect Intruder ---

    elif intruder_nav == "Detect Intruder":
            st.markdown(
                "<div style='text-align:center; margin-bottom: 1em; font-size:1.2em; color:#00b894; "
                "font-weight:700; letter-spacing:0.5px;'>"
                "ℹ️ <b>Check the threshold confidence value on the left side to adjust detection sensitivity.</b>"
                "</div>",
                unsafe_allow_html=True
            )
            st.subheader("Live Intruder Detection")
            
            threshold = st.sidebar.slider("Recognition Confidence Threshold (lower is stricter)", 50, 120, 85, 1)

            class LBPHIntruderDetector(VideoProcessorBase):
                found = False
                confidence_value = None
                last_snapshot_time = 0


                def __init__(self):
                    self.recognizer = load_intruder_model()
                    self.face_cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                    )

                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                    LBPHIntruderDetector.found = False
                    LBPHIntruderDetector.confidence_value = None
                    for (x, y, w, h) in faces:
                        face_roi = gray[y:y+h, x:x+w]
                        face_roi_resized = cv2.resize(face_roi, FACE_SIZE)
                        label, confidence = self.recognizer.predict(face_roi_resized)
                        LBPHIntruderDetector.confidence_value = confidence
                        if label == 0 and confidence < threshold:
                            LBPHIntruderDetector.found = True
                            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
                            cv2.putText(img, f"Intruder ({confidence:.1f})", (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            now_time = time.time()
                            if now_time - LBPHIntruderDetector.last_snapshot_time > 2:
                                save_snapshot(img, is_intruder=True)
                                LBPHIntruderDetector.last_snapshot_time = now_time
                        else:
                            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(img, f"Not Intruder ({confidence:.1f})", (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            now_time = time.time()
                            if now_time - LBPHIntruderDetector.last_snapshot_time > 3:
                                save_snapshot(img, is_intruder=False)
                                LBPHIntruderDetector.last_snapshot_time = now_time


                    border_color = (0, 0, 255) if LBPHIntruderDetector.found else (0, 255, 0)
                    return av.VideoFrame.from_ndarray(
                        cv2.copyMakeBorder(img, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=border_color),
                        format="bgr24"
                    )

            status = st.empty()
            ctx = webrtc_streamer(
                key="intruder",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=LBPHIntruderDetector,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            if ctx.state.playing:
                if LBPHIntruderDetector.found:
                    status.error(f"🚨 Intruder Found! (Confidence: {LBPHIntruderDetector.confidence_value:.1f})")
                else:
                    if LBPHIntruderDetector.confidence_value is not None:
                        status.info(f"Scanning... (Last confidence: {LBPHIntruderDetector.confidence_value:.1f})")
                    else:
                        status.info("Scanning for intruder...")
            
            st.markdown("---")
            st.subheader("Recent Intruder Snapshots")
            snapshot_paths = get_snapshot_paths(is_intruder=True)
            if snapshot_paths:
                cols = st.columns(min(4, len(snapshot_paths)))
                for i, path in enumerate(snapshot_paths):
                    with cols[i % len(cols)]:
                        st.image(path, use_container_width=True, caption="Intruder")
            else:
                st.write("No intruder snapshots yet.")


            st.info("Tip: Register multiple images of the intruder with different angles/lighting for best results.")

    # --- Intruder Gallery ---
    elif intruder_nav == "Gallery":
        st.subheader("Intruder Gallery")
        gallery_tabs = st.tabs(["Intruder", "Not Intruder"])

        with gallery_tabs[0]:
            st.markdown("#### Intruder Snapshots")
            snapshot_paths = get_snapshot_paths(is_intruder=True)
            if snapshot_paths:
                cols = st.columns(min(4, len(snapshot_paths)))
                for i, path in enumerate(snapshot_paths):
                    with cols[i % len(cols)]:
                        st.image(path, use_container_width=True)
                if st.button("Clear Intruder Snapshots", key="clear_intruder"):
                    clear_snapshots(is_intruder=True)
                    st.success("All intruder snapshots cleared.")
            else:
                st.write("No intruder snapshots yet.")

        with gallery_tabs[1]:
            st.markdown("#### Not Intruder Snapshots")
            not_intruder_paths = get_snapshot_paths(is_intruder=False)
            if not_intruder_paths:
                cols = st.columns(min(4, len(not_intruder_paths)))
                for i, path in enumerate(not_intruder_paths):
                    with cols[i % len(cols)]:
                        st.image(path, use_container_width=True)
                if st.button("Clear Not Intruder Snapshots", key="clear_not_intruder"):
                    clear_snapshots(is_intruder=False)
                    st.success("All not intruder snapshots cleared.")
            else:
                st.write("No not-intruder snapshots yet.")


# --- Age and Gender Detection Page ---
elif page == "Intruder Age and Gender":
    st.title("Intruder Age & Gender Detection")
    st.caption("Analyze all saved intruder snapshots and estimate age and gender for each detected face.")
    snapshot_paths = get_snapshot_paths(is_intruder=True)
    if not snapshot_paths:
        st.warning("No intruder snapshots found. Please detect an intruder first.")
        st.stop()
    st.subheader("Analyzing Snapshots...")
    age_net, gender_net = load_age_gender_models()
    results = []
    for path in snapshot_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]
            try:
                face_img_resized = cv2.resize(face_img, (227, 227))
                gender, age = predict_age_gender(face_img_resized, age_net, gender_net)
                results.append((path, gender, age))
            except Exception:
                continue
    if results:
        st.success(f"Detected {len(results)} intruder face(s) in snapshots.")
        for path, gender, age in results:
                basename = os.path.basename(path)
                # Extract timestamp from filename
                try:
                    dt_str = basename.split("_")[1] + " " + basename.split("_")[2][:6]
                    dt_fmt = datetime.strptime(dt_str, "%Y%m%d %H%M%S")
                    dt_display = dt_fmt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    dt_display = "Unknown"
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(path, use_container_width=True)
                with col2:
                    st.markdown(
                        f"**Date/Time:** {dt_display}  \n"
                        f"**Gender:** {gender}  \n"
                        f"**Age Range:** {age}",
                        unsafe_allow_html=True
                    )   
    else:
        st.warning("No faces detected in snapshots for age/gender analysis.")
    if st.button("Clear Intruder Snapshots", key="clear_intruder_age_gender"):
        clear_snapshots(is_intruder=True)
        st.success("All intruder snapshots cleared.")
    st.info("Tip: Age and gender detection is based on deep learning models and may not be 100% accurate.")


# ---- Footer ----
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color:#a6b0c3; font-size:0.95rem; margin-top:2em;'>"
    "© 2025 VisionSentinel. All rights reserved."
    "</div>", unsafe_allow_html=True
)