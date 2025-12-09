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
API_BASE_URL = os.getenv("GARLIC_API_URL", "http://localhost:8000")


def api_post(endpoint: str, files=None, json=None) -> dict:
    url = f"{API_BASE_URL}{endpoint}"
    response = requests.post(url, files=files, json=json, timeout=60)
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


def upload_image_to_api(image_bytes: bytes, filename: str) -> tuple[str, str]:
    files = {"file": (filename, image_bytes)}
    data = api_post("/api/upload", files=files)
    return data["image_id"], data["status"]


def start_processing(image_id: str) -> dict:
    return api_post(f"/api/process/{image_id}")


def fetch_result(image_id: str) -> bytes:
    data = api_get(f"/api/result/{image_id}")
    return base64.b64decode(data["annotated_image_base64"])


def submit_feedback(image_id: str, decision: str) -> dict:
    payload = {"image_id": image_id, "decision": decision}
    return api_post("/api/feedback", json=payload)


def step_indicator(current_step: int) -> None:
    steps = [
        "1. Capture or Upload",
        "2. Send to AI",
        "3. Review Result",
        "4. Provide Feedback",
    ]
    cols = st.columns(len(steps))
    for idx, label in enumerate(steps, start=1):
        with cols[idx - 1]:
            status = "✅" if current_step > idx else ("🟡" if current_step == idx else "⚪️")
            st.markdown(f"{status} **{label}**")


def show_step_one() -> Optional[bytes]:
    st.subheader("Step 1 · Capture or upload a garlic image")
    st.write("Use your camera or upload an existing picture of garlic for the AI to inspect.")

    camera_image = st.camera_input("Capture using camera", key="camera_input")
    uploaded_file = st.file_uploader(
        "Or upload an image file (PNG, JPG)", type=["png", "jpg", "jpeg"], key="file_uploader"
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
        st.image(image_bytes, caption="Selected image", use_column_width=True)
        st.session_state["image_preview"] = image_bytes
        st.session_state["image_filename"] = filename

    return image_bytes


def handle_step_two() -> None:
    st.subheader("Step 2 · Send to the AI model")
    st.write("We are processing your image. This may take a few seconds.")

    progress = st.progress(0)
    for pct in range(0, 101, 5):
        time.sleep(0.05)
        progress.progress(pct)

    process_response = start_processing(st.session_state["image_id"])
    st.session_state["processing_status"] = process_response["status"]

    if process_response["status"] == "completed":
        st.success("Processing finished. Moving to results.")
        st.session_state["step"] = 3
        st.rerun()
    else:
        st.info(process_response["message"])


def handle_step_three() -> None:
    st.subheader("Step 3 · Review the AI output")
    if st.session_state["annotated_image"] is None:
        annotated_bytes = fetch_result(st.session_state["image_id"])
        st.session_state["annotated_image"] = annotated_bytes

    st.image(
        st.session_state["annotated_image"],
        caption="AI annotated garlic detection",
        use_column_width=True,
    )
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


def main() -> None:
    st.set_page_config(page_title="Garlic AI Dashboard", page_icon="🧄", layout="wide")
    st.title("Garlic AI Workflow Dashboard")
    st.caption("Guide your garlic images through the AI detection pipeline.")

    init_session_state()
    
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
    step_indicator(st.session_state["step"])

    if st.session_state["step"] == 1:
        image_bytes = show_step_one()
        if image_bytes and st.button("Send to AI", type="primary"):
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
        handle_step_four()


if __name__ == "__main__":
    main()

