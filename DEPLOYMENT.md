# Deploying to Streamlit Cloud

## Setup Steps

### 1. Prepare Your Local Environment
✅ All fixes applied:
- `requirements.txt` encoding fixed (UTF-8)
- `.streamlit/secrets.toml` created for local development
- `.streamlit/config.toml` created for optimal settings
- `.gitignore` updated to exclude sensitive files

### 2. Deploy to Streamlit Cloud

1. **Push to GitHub** (make sure .env and .streamlit/secrets.toml are in .gitignore):
   ```bash
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   git push
   ```

2. **Connect to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your GitHub repo
   - Set the branch to `main` (or your branch)
   - Set the main file path to `app.py`
   - Click "Deploy"

3. **Add Secrets on Streamlit Cloud**:
   - After deployment, click the **"Deploy"** menu → **"Settings"**
   - Go to **"Secrets"** tab
   - Add your secrets as:
     ```toml
     GROQ_API_KEY = "your-actual-groq-api-key"
     ```
   - Save - the app will automatically restart

### 3. Common Issues & Fixes

#### ❌ "ModuleNotFoundError" or import errors
- All dependencies are now in `requirements.txt`
- Streamlit Cloud will auto-install from `requirements.txt`

#### ❌ "API key not found" error
- Go to app settings → **Secrets** tab
- Add `GROQ_API_KEY` there (NOT in code or .env)

#### ❌ App crashes or times out
- Streamlit Cloud has resource limits
- If the app is memory-intensive, consider:
  - Caching results with `@st.cache_data`
  - Breaking down heavy computations

### 4. Local Testing Before Deployment

To test with Streamlit Cloud's secrets locally:
```bash
# Fill in .streamlit/secrets.toml with your actual API key
# Then run:
streamlit run app.py
```

## Environment Variables

This app needs:
- `GROQ_API_KEY` - Your Groq API key

All other dependencies are handled by `requirements.txt`.
