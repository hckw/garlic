# Testing Locally

## Prerequisites

1. Make sure you have a `.env` file in the root directory with your Roboflow credentials:
   ```
   ROBOFLOW_API_KEY=your_api_key_here
   ROBOFLOW_MODEL_ID=garlic-2-wvv7b/10
   GARLIC_CONFIDENCE_THRESHOLD=0.5
   ```

2. Install dependencies (if not already installed):
   ```bash
   pip install -r requirements-backend.txt
   pip install -r requirements-frontend.txt
   ```

## Step 1: Start the Backend

Open Terminal 1:

```bash
cd /Users/alfinaaura/garlic-fe

# Load environment variables from .env
set -a; source .env; set +a

# Start the FastAPI backend server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

Keep this terminal running!

## Step 2: Start the Frontend

Open Terminal 2 (new terminal window):

```bash
cd /Users/alfinaaura/garlic-fe

# Start the Streamlit frontend
streamlit run frontend/app.py
```

You should see:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

The browser should automatically open to `http://localhost:8501`

## Step 3: Test the Application

### Test Automatic Mode

1. When the app loads, you'll see a mode selection screen
2. Click **"🤖 Automatic Mode"**
3. Upload an image or take a picture
4. Click "Send to AI"
5. Wait for processing to complete
6. The annotated result will appear
7. After 5 seconds, it will automatically return to Step 1
8. The cycle continues - upload another image to test the loop

### Test Feedback Mode

1. Click **"🔄 Change Mode"** in the sidebar
2. Select **"💬 Feedback Mode"**
3. Upload an image or take a picture
4. Click "Send to AI"
5. Wait for processing
6. Review the result
7. Click "Continue to feedback"
8. You'll see Accept/Reject buttons
9. Test the feedback workflow

## Troubleshooting

### Backend won't start
- Check that your `.env` file exists and has correct values
- Make sure port 8000 is not in use: `lsof -i :8000`
- Kill process if needed: `kill -9 <PID>`

### Frontend can't connect to backend
- Verify backend is running on `http://localhost:8000`
- Check the backend logs for errors
- The frontend defaults to `http://localhost:8000` automatically

### Environment variables not loading
- Use: `set -a; source .env; set +a` before starting uvicorn
- Or manually export: `export ROBOFLOW_API_KEY=your_key`

## Stopping the Servers

- Press `Ctrl+C` in each terminal to stop the servers
- Stop backend first, then frontend

