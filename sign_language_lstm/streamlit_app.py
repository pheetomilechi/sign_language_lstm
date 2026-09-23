"""Browser-based live demo for continuous sign-language recognition."""

import os
import sys

if __package__ in (None, ""):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from sign_language_lstm.config import MODEL_PATH
    from sign_language_lstm.mp_utils import new_holistic_model
    from sign_language_lstm.recognizer import (
        ContinuousRecognizer,
        ensure_model_files_exist,
        load_label_map_inverse,
    )
else:
    from .config import MODEL_PATH
    from .mp_utils import new_holistic_model
    from .recognizer import (
        ContinuousRecognizer,
        ensure_model_files_exist,
        load_label_map_inverse,
    )

import av
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from tensorflow.keras.models import load_model


st.set_page_config(page_title="Sign Language LSTM",
                   page_icon="🤟", layout="wide")


def build_rtc_configuration():
    """Build a WebRTC ICE config from environment variables.

    The app supports a comma-separated STUN list and optional TURN credentials.
    This is important when the browser is behind NAT/firewalls or running in a
    remote environment such as Codespaces, a VPS, or a cloud VM.
    """
    ice_servers = []

    stun_servers = os.getenv("STUN_SERVERS")
    if stun_servers:
        for server in stun_servers.split(","):
            cleaned = server.strip()
            if cleaned:
                ice_servers.append({"urls": [cleaned]})
    else:
        ice_servers.append({"urls": ["stun:stun.l.google.com:19302"]})

    turn_url = os.getenv("TURN_URL")
    if turn_url:
        turn_username = os.getenv("TURN_USERNAME", "")
        turn_credential = os.getenv("TURN_CREDENTIAL", "")
        ice_servers.append({
            "urls": [turn_url],
            "username": turn_username,
            "credential": turn_credential,
        })

    return {"iceServers": ice_servers}


@st.cache_resource(show_spinner="Loading the sign language model...")
def load_recognition_assets():
    ensure_model_files_exist()
    return load_model(MODEL_PATH), load_label_map_inverse()


class SignLanguageVideoProcessor(VideoProcessorBase):
    def __init__(self, model, inverse_label_map):
        self.recognizer = ContinuousRecognizer(model, inverse_label_map)
        self.holistic = new_holistic_model(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def recv(self, frame):
        video_frame = frame.to_ndarray(format="bgr24")
        processed_frame = self.recognizer.process_frame(
            video_frame, self.holistic)
        cv2.rectangle(
            processed_frame,
            (0, 0),
            (processed_frame.shape[1], 64),
            (24, 31, 39),
            -1,
        )
        cv2.putText(
            processed_frame,
            " ".join(self.recognizer.sentence) or "Ready",
            (16, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")


def main():
    st.title("Continuous Sign Language Recognition")
    st.caption("Use the webcam to build a sentence from recognized signs.")

    try:
        model, inverse_label_map = load_recognition_assets()
    except FileNotFoundError as error:
        st.error(str(error))
        st.info(
            "Train the model locally and push the generated model files to GitHub before deploying. "
            "The app needs the files in sign_language_lstm/models/ to be present in the repo."
        )
        return

    ctx = webrtc_streamer(
        key="sign-language-demo",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: SignLanguageVideoProcessor(
            model, inverse_label_map),
        rtc_configuration=build_rtc_configuration(),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    controls = st.columns([1, 1, 3])
    with controls[0]:
        if st.button("Clear sentence", use_container_width=True):
            if ctx.video_processor:
                ctx.video_processor.recognizer.reset()
            st.rerun()
    with controls[1]:
        recording = st.session_state.get("recording", True)
        label = "Stop recording" if recording else "Record"
        if st.button(label, use_container_width=True):
            recording = not recording
            st.session_state.recording = recording
        if ctx.video_processor:
            ctx.video_processor.recognizer.set_recording(recording)

    sentence_placeholder = st.empty()
    status_placeholder = st.empty()
    if ctx.video_processor:
        recognizer = ctx.video_processor.recognizer
        sentence_placeholder.markdown(
            "### " + (" ".join(recognizer.sentence)
                      or "Your sentence will appear here")
        )
        status = "Recording" if recognizer.recording else "Paused"
        status_placeholder.caption(
            f"{status} | Latest sign: {recognizer.last_word or 'none'}")
    else:
        sentence_placeholder.markdown("### Start the camera to begin")


if __name__ == "__main__":
    main()
