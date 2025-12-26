# Garlic Detection Dashboard Plan

## Goals
- Provide streamlined 4-step workflow for uploading garlic images, sending them to an AI model, viewing annotated outputs, and giving feedback.
- Build FastAPI backend to expose upload, processing, annotation retrieval, and feedback endpoints.
- Build Streamlit frontend to orchestrate the workflow with progress indication and rejection handling rules.

## High-Level Flow
1. **Capture / Upload**: User supplies image via camera or file upload and sends it to backend `/api/upload`. Backend stores the image and returns an `image_id`.
2. **Process**: Streamlit triggers `/api/process/{image_id}`. Backend simulates model processing (placeholder) and produces annotated image artifact.
3. **Review Output**: Streamlit retrieves `/api/result/{image_id}` to display annotated garlic image to user.
4. **Feedback**: Streamlit calls `/api/feedback` with Accept/Reject. Backend tracks reject count per `image_id` and instructs UI to either reprocess (<=2 rejects) or reset to Step 1 (>2 rejects) while showing warning popup.

## Backend Components
- `backend/main.py`: FastAPI app with CORS, in-memory `ImageStore` that tracks metadata.
- Storage folders: `data/uploads` and `data/annotated` for original and annotated images. Annotated output uses Pillow overlay placeholder.
- Endpoints:
  - `POST /api/upload` -> returns `image_id`.
  - `POST /api/process/{image_id}` -> simulates processing, updates status.
  - `GET /api/result/{image_id}` -> returns annotated image as base64 + status.
  - `POST /api/feedback` -> tracks rejections/acceptances and returns actions `reprocess` or `reset`.

## Frontend Components
- `frontend/app.py` (Streamlit): manages `st.session_state` for `step`, `image_id`, `reject_count`.
- Step components encapsulated in helper functions for clarity.
- Uses `requests` to call backend and `st.progress` for Step 2 progress bar.
- Implements popup-like warning via `st.dialog` (Streamlit >=1.28) triggered when backend instructs `reset` after >2 rejects.

## Future Backend Integration Points
- Replace simulated annotation in `/api/process` with actual AI pipeline invocation.
- Persist metadata in database / object storage (S3, etc.) for production.
- Enhance security (auth, rate limiting) when hooking to real backend.

## Deployment Notes
- Single `requirements.txt` for both FastAPI & Streamlit during dev.
- Run backend: `uvicorn backend.main:app --reload`.
- Run frontend: `streamlit run frontend/app.py`.
