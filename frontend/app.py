from __future__ import annotations

import base64
import os
import time
from typing import Optional

import requests
import streamlit as st


# Get API URL from environment variable
# For local development: defaults to http://localhost:8000
# For production: MUST be set to your Railway backend URL in Streamlit Cloud settings
_api_url = os.getenv("GARLIC_API_URL", "http://localhost:8000")
# Remove trailing slash to avoid double slashes in URLs
API_BASE_URL = _api_url.rstrip("/")


def api_post(endpoint: str, files=None, json=None, params=None) -> dict:
    url = f"{API_BASE_URL}{endpoint}"
    # Process endpoint returns immediately, so short timeout is fine
    response = requests.post(url, files=files, json=json, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def api_get(endpoint: str) -> dict:
    url = f"{API_BASE_URL}{endpoint}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def init_session_state() -> None:
    defaults = {
        "step": 1,
        "image_id": None,
        "image_preview": None,
        "image_filename": None,
        "annotated_image": None,
        "processing_status": None,
        "reject_count": 0,
        "last_feedback_message": "",
        "reject_popup_message": "",
        "reject_popup_timestamp": None,
        "mode": None,  # "automatic" or "feedback" - None means not selected yet
        "mode_selected": False,
        "auto_mode_timer_start": None,  # When to start 5-second countdown in automatic mode
        "step1_reset_counter": 0,  # Counter to reset camera/file uploader widgets
        "confidence_threshold": 0.5,  # User-adjustable confidence threshold
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_to_step(step: int = 1) -> None:
    st.session_state.update(
        {
            "step": step,
            "image_id": None if step == 1 else st.session_state.get("image_id"),
            "image_preview": None if step == 1 else st.session_state.get("image_preview"),
            "image_filename": None if step == 1 else st.session_state.get("image_filename"),
            "annotated_image": None,
            "processing_status": None,
            "last_feedback_message": "",
        }
    )
    if step == 1:
        st.session_state["reject_count"] = 0
        # Increment counter to reset camera/file uploader widgets
        st.session_state["step1_reset_counter"] = st.session_state.get("step1_reset_counter", 0) + 1
        # Clear all image-related state when returning to step 1
        st.session_state["image_preview"] = None
        st.session_state["image_filename"] = None
        st.session_state["image_id"] = None
        st.session_state["processing_started"] = None
        st.session_state["processing_status"] = None


def upload_image_to_api(image_bytes: bytes, filename: str) -> tuple[str, str]:
    files = {"file": (filename, image_bytes)}
    data = api_post("/api/upload", files=files)
    return data["image_id"], data["status"]


def start_processing(image_id: str, confidence_threshold: float = 0.5) -> dict:
    # Pass confidence threshold as query parameter
    return api_post(f"/api/process/{image_id}", params={"confidence_threshold": confidence_threshold})


def fetch_result(image_id: str) -> bytes:
    data = api_get(f"/api/result/{image_id}")
    return base64.b64decode(data["annotated_image_base64"])


def submit_feedback(image_id: str, decision: str) -> dict:
    payload = {"image_id": image_id, "decision": decision}
    return api_post("/api/feedback", json=payload)


def step_indicator(current_step: int, mode: str = "feedback") -> None:
    steps = [
        "1. Capture or Upload",
        "2. Send to AI",
        "3. Review Result",
    ]
    # Only show step 4 in feedback mode
    if mode == "feedback":
        steps.append("4. Provide Feedback")
    
    cols = st.columns(len(steps))
    for idx, label in enumerate(steps, start=1):
        with cols[idx - 1]:
            status = "✅" if current_step > idx else ("🟡" if current_step == idx else "⚪️")
            st.markdown(f"{status} **{label}**")


def show_step_one() -> Optional[bytes]:
    st.subheader("Step 1 · Capture or upload a garlic image")
    st.write("Use your camera or upload an existing picture of garlic for the AI to inspect.")

    # Use counter in key to reset widgets when returning to step 1
    reset_counter = st.session_state.get("step1_reset_counter", 0)
    camera_image = st.camera_input("Capture using camera", key=f"camera_input_{reset_counter}")
    uploaded_file = st.file_uploader(
        "Or upload an image file (PNG, JPG)", type=["png", "jpg", "jpeg"], key=f"file_uploader_{reset_counter}"
    )

    image_bytes: Optional[bytes] = None
    filename = "captured_image.png"

    if camera_image is not None:
        image_bytes = camera_image.getvalue()
        filename = camera_image.name or filename
    elif uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name

    if image_bytes:
        # Display image in a smaller, centered column
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            st.image(image_bytes, caption="Selected image", use_column_width=True)
        st.session_state["image_preview"] = image_bytes
        st.session_state["image_filename"] = filename

    return image_bytes


def handle_step_two() -> None:
    st.subheader("Step 2 · Send to the AI model")
    st.write("We are processing your image. This may take maximum a minute.")

    # Start processing (returns immediately)
    if st.session_state.get("processing_started") is None:
        try:
            confidence_threshold = st.session_state.get("confidence_threshold", 0.5)
            process_response = start_processing(st.session_state["image_id"], confidence_threshold)
            st.session_state["processing_status"] = process_response["status"]
            st.session_state["processing_started"] = True
            st.info(f"Processing started with confidence threshold: {confidence_threshold:.2f}. Waiting for completion...")
        except requests.RequestException as exc:
            st.error(f"Failed to start processing: {exc}")
            return

    # Poll for completion
    progress = st.progress(0)
    status_text = st.empty()
    
    max_attempts = 60  # Poll for up to 5 minutes (60 * 5 seconds)
    poll_interval = 5  # Check every 5 seconds
    
    for attempt in range(max_attempts):
        try:
            # Check result endpoint to see if processing is complete
            result = api_get(f"/api/result/{st.session_state['image_id']}")
            
            if result["status"] == "completed":
                progress.progress(100)
                status_text.success("✅ Processing completed!")
                st.session_state["processing_status"] = "completed"
                st.session_state["step"] = 3
                st.session_state["processing_started"] = None  # Reset for next time
                time.sleep(1)  # Brief pause to show success message
                st.rerun()
                return
            elif result["status"] == "processing":
                # Still processing, update progress
                progress_pct = min(90, 10 + (attempt * 2))  # Progress from 10% to 90%
                progress.progress(progress_pct)
                status_text.info(f"⏳ Processing... (attempt {attempt + 1}/{max_attempts})")
            else:
                # Error or other status
                status_text.warning(f"Status: {result['status']}")
        except requests.HTTPError as e:
            if e.response.status_code == 400:
                # Not ready yet (400 = "Image is not ready yet")
                progress_pct = min(90, 10 + (attempt * 2))
                progress.progress(progress_pct)
                status_text.info(f"⏳ Processing... (attempt {attempt + 1}/{max_attempts})")
            else:
                status_text.error(f"Error checking status: {e}")
                break
        except requests.RequestException as exc:
            status_text.error(f"Connection error: {exc}")
            break
        
        time.sleep(poll_interval)
    
    # If we get here, we've exhausted attempts
    progress.progress(100)
    status_text.error("⏱️ Processing is taking longer than expected. Please refresh and try again.")


def handle_step_three() -> None:
    st.subheader("Step 3 · Review the AI output")
    if st.session_state["annotated_image"] is None:
        annotated_bytes = fetch_result(st.session_state["image_id"])
        st.session_state["annotated_image"] = annotated_bytes

    # Display annotated image in a smaller, centered column
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.image(
            st.session_state["annotated_image"],
            caption="AI annotated garlic detection",
            use_column_width=True,
        )
    
    mode = st.session_state.get("mode", "feedback")
    
    if mode == "automatic":
        # Automatic mode: show for 5 seconds then auto-return to step 1
        if st.session_state.get("auto_mode_timer_start") is None:
            st.session_state["auto_mode_timer_start"] = time.time()
            st.info("🤖 Automatic mode: Result will be shown for 5 seconds, then returning to capture new image...")
            # Trigger rerun to start the countdown
            time.sleep(0.1)
            st.rerun()
        else:
            elapsed = time.time() - st.session_state["auto_mode_timer_start"]
            remaining = max(0, 5 - elapsed)
            if remaining > 0:
                st.info(f"🤖 Automatic mode: Returning to capture in {int(remaining)} seconds...")
                time.sleep(0.5)  # Update display every 0.5 seconds
                st.rerun()
            else:
                # 5 seconds passed, return to step 1
                st.session_state["auto_mode_timer_start"] = None
                st.session_state["annotated_image"] = None
                reset_to_step(1)
                st.rerun()
    else:
        # Feedback mode: show button to continue to feedback
        st.info("Inspect the annotations carefully before sending your feedback.")
        if st.button("Continue to feedback", type="primary"):
            st.session_state["step"] = 4
            st.rerun()


def handle_step_four() -> None:
    st.subheader("Step 4 · Provide feedback")

    cols = st.columns(2)
    with cols[0]:
        if st.button("✅ Accept", use_container_width=True):
            response = submit_feedback(st.session_state["image_id"], "accept")
            st.session_state["last_feedback_message"] = response.get("message", "")
            st.success(st.session_state["last_feedback_message"])
            reset_to_step(1)
            st.rerun()

    with cols[1]:
        if st.button("❌ Reject & Reprocess", type="secondary", use_container_width=True):
            response = submit_feedback(st.session_state["image_id"], "reject")
            st.session_state["reject_count"] = response["reject_count"]
            st.session_state["last_feedback_message"] = response.get("message", "")

            if response["action"] == "reprocess":
                st.info(st.session_state["last_feedback_message"])
                st.session_state["annotated_image"] = None
                st.session_state["step"] = 2
                st.rerun()
            else:
                st.warning(st.session_state["last_feedback_message"])
                st.session_state["reject_popup_message"] = response["message"]
                st.session_state["reject_popup_timestamp"] = time.time()
                reset_to_step(1)
                st.rerun()

    if st.session_state["last_feedback_message"]:
        st.caption(st.session_state["last_feedback_message"])


def show_mode_selection() -> None:
    """Show mode selection dialog before starting the workflow."""
    if st.session_state.get("mode_selected", False):
        return
    
    # Confidence threshold slider (moved to top)
    st.markdown("**Confidence Threshold:**")
    confidence_threshold = st.slider(
        "Adjust the confidence threshold (0.0 - 1.0). Lower values detect more objects but may include false positives.",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get("confidence_threshold", 0.5),
        step=0.05,
        key="mode_selection_threshold"
    )
    st.session_state["confidence_threshold"] = confidence_threshold
    
    # Apply color to slider based on threshold (green above 70%, red below)
    if confidence_threshold > 0.7:
        slider_color = "#28a745"  # Green
    else:
        slider_color = "#dc3545"  # Red
    
    st.markdown(
        f"""
        <style>
        /* Target the StyledThumbValue class for the value number above the slider */
        .StyledThumbValue {{
            color: {slider_color} !important;
        }}
        /* Also target with emotion cache classes */
        .st-emotion-cache-10y5sf6 {{
            color: {slider_color} !important;
        }}
        .ew7r33m2 {{
            color: {slider_color} !important;
        }}
        /* Keep stTickBarMin in default color (do not change) */
        .stTickBarMin {{
            color: inherit !important;
            background-color: inherit !important;
        }}
        /* Target the slider track filled portion */
        div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div:first-child {{
            background: linear-gradient(to right, {slider_color} 0%, {slider_color} {confidence_threshold * 100}%, #d3d3d3 {confidence_threshold * 100}%, #d3d3d3 100%) !important;
        }}
        /* Target the slider thumb/handle */
        div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div > div {{
            background-color: {slider_color} !important;
            border-color: {slider_color} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    
    st.info("**Select your workflow mode:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("**Automatic Mode**", use_container_width=True, type="primary"):
            st.session_state["mode"] = "automatic"
            st.session_state["mode_selected"] = True
            st.rerun()
        st.caption("Images are processed automatically and shown for 5 seconds before capturing the next image. No feedback step.")
    
    with col2:
        if st.button("**Feedback Mode**", use_container_width=True):
            st.session_state["mode"] = "feedback"
            st.session_state["mode_selected"] = True
            st.rerun()
        st.caption("Standard workflow with manual feedback. You can accept or reject results.")


def main() -> None:
    st.set_page_config(page_title="Garlic AI Dashboard", page_icon="🧄", layout="wide")
    st.title("Garlic AI Workflow Dashboard")
    st.caption("Guide your garlic images through the AI detection pipeline.")

    init_session_state()
    
    # Show mode selection if not selected yet
    if not st.session_state.get("mode_selected", False):
        show_mode_selection()
        return  # Don't proceed until mode is selected
    
    # Show popup message for 10 seconds
    if st.session_state.get("reject_popup_message") and st.session_state.get("reject_popup_timestamp"):
        current_time = time.time()
        elapsed = current_time - st.session_state["reject_popup_timestamp"]
        
        if elapsed < 10:
            # Show the popup message with remaining time and close button
            remaining = int(10 - elapsed)
            st.error(
                f"{st.session_state['reject_popup_message']} (Auto-closing in {remaining}s)",
                icon="⚠️"
            )
            # Add close button
            if st.button("Close", key="close_popup"):
                st.session_state["reject_popup_message"] = ""
                st.session_state["reject_popup_timestamp"] = None
                st.rerun()
            else:
                # Rerun every second to update the countdown
                time.sleep(1)
                st.rerun()
        else:
            # 10 seconds passed, clear the message
            st.session_state["reject_popup_message"] = ""
            st.session_state["reject_popup_timestamp"] = None
    
    # Show current mode indicator
    mode = st.session_state.get("mode", "feedback")
    mode_label = "Automatic Mode" if mode == "automatic" else "Feedback Mode"
    st.sidebar.info(f"**Current Mode:** {mode_label}")
    if st.sidebar.button("Switch mode"):
        st.session_state["mode_selected"] = False
        st.session_state["mode"] = None
        st.session_state["confidence_threshold"] = 0.5  # Reset threshold when switching modes
        reset_to_step(1)
        st.rerun()
    
    step_indicator(st.session_state["step"], mode=mode)

    if st.session_state["step"] == 1:
        image_bytes = show_step_one()
        
        # In automatic mode, immediately send to AI when image is captured/uploaded
        # Only upload if we haven't already started processing this image
        if mode == "automatic" and image_bytes and st.session_state.get("image_id") is None:
            with st.spinner("Uploading image..."):
                try:
                    image_id, status = upload_image_to_api(
                        image_bytes, st.session_state["image_filename"] or "garlic.png"
                    )
                    st.session_state["image_id"] = image_id
                    st.session_state["processing_status"] = status
                    st.session_state["step"] = 2
                    st.rerun()
                except requests.RequestException as exc:
                    st.error(f"Upload failed: {exc}")
        # In feedback mode, show button to manually send to AI
        elif mode == "feedback" and image_bytes and st.button("Send to AI", type="primary"):
            with st.spinner("Uploading image..."):
                try:
                    image_id, status = upload_image_to_api(
                        image_bytes, st.session_state["image_filename"] or "garlic.png"
                    )
                    st.session_state["image_id"] = image_id
                    st.session_state["processing_status"] = status
                    st.session_state["step"] = 2
                    st.success("Image uploaded. Moving to processing.")
                    st.rerun()
                except requests.RequestException as exc:
                    st.error(f"Upload failed: {exc}")
    elif st.session_state["step"] == 2:
        try:
            handle_step_two()
        except requests.RequestException as exc:
            st.error(f"Processing failed: {exc}")
    elif st.session_state["step"] == 3:
        try:
            handle_step_three()
        except requests.RequestException as exc:
            st.error(f"Unable to fetch result: {exc}")
    elif st.session_state["step"] == 4:
        # Only show step 4 in feedback mode
        current_mode = st.session_state.get("mode", "feedback")
        if current_mode == "feedback":
            handle_step_four()
        else:
            # Shouldn't reach here in automatic mode, but handle gracefully
            reset_to_step(1)
            st.rerun()


if __name__ == "__main__":
    main()

