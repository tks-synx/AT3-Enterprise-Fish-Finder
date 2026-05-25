# Setting Up TinyLlama + Ollama (Local AI)

## Step 1: Install Ollama
1. Go to https://ollama.ai
2. Download for macOS (M1 compatible)
3. Install and open the app

## Step 2: Pull TinyLlama Model
Open Terminal and run:
```bash
ollama pull tinyllama
```
This downloads the ~1.1GB model (one-time only)

## Step 3: Verify It's Running
The Ollama app should run in the background automatically. To test:
```bash
curl http://localhost:11434/api/tags
```

You should see `tinyllama` listed.

## Step 4: Install Python Package
In your project directory, run:
```bash
pip install ollama
```

## Step 5: Start Using the AI

Run your Streamlit app with AI enabled:
```bash
python3 -m streamlit run BasicProduction/fish_app.py
```

The AI will appear as a chat sidebar in your app. You can ask it questions about your fish data!

---

## Troubleshooting

**Ollama not starting?**
- Make sure the Ollama app is running (check your Mac menu bar)

**Model not loading?**
- Run `ollama list` to see if tinyllama is installed
- If not, run `ollama pull tinyllama` again

**Port 11434 in use?**
- Kill the process: `lsof -i :11434` then `kill -9 <PID>`
