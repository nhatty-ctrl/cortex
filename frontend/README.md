<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://ai.google.dev/static/site-assets/images/share-ais-513315318.png" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/56ff8d90-26ea-474d-ba2a-77da1c7502b5

## Run Locally

**Prerequisites:** Node.js

1. Install dependencies:
   `npm install`
2. Create `.env.local` from `.env.example` and set your `GEMINI_API_KEY`.
3. Run the app:
   `npm run dev`

## Backend Integration

This frontend is intentionally separate from the backend. Run the backend from the `cortex_ai_final/backbone_ai` folder, then point this frontend to it using `VITE_API_URL`.

### Backend local startup

```bash
cd ../cortex_ai_final/backbone_ai
pip install -r requirements.txt
playwright install chromium
cp .env.example .env      # paste your keys
uvicorn main:app --reload --port 8000
```

The backend will be live at `http://localhost:8000`.

### Frontend configuration

In the frontend folder:

```bash
npm install
cp .env.example .env
```

Then set:

```env
VITE_API_URL="http://localhost:8000"
```

If `VITE_API_URL` is not set, the frontend defaults to `http://localhost:8000`.

### Start the frontend

```bash
npm run dev
```

When deployed, update `VITE_API_URL` to the backend public URL.
