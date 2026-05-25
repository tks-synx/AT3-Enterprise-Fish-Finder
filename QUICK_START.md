## python3 -m streamlit run BasicProduction/fish_app.py


# Quick Start: TinyLlama AI + Fisheries Heatmap

## 🚀 Get Started in 5 Minutes

### Step 1: Install Ollama (One-time setup)
1. Go to https://ollama.ai
2. Download for **macOS**
3. Open the app (keep it running in background)

### Step 2: Download TinyLlama Model
Open Terminal in your project folder and run:
```bash
ollama pull tinyllama
```
Wait for it to finish (~5 mins on first run)

### Step 3: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the App
```bash
python3 -m streamlit run BasicProduction/fish_app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 💬 Using the AI

After generating a map, look for the **"🤖 Fish AI Assistant"** panel on the right sidebar.

**You can ask things like:**
- "Where are the most active fishing zones?"
- "Compare tuna patterns between summer and winter"
- "Which species are found in the northern regions?"
- "Summarize the current data"
- "What time of year has the most activity?"

**Quick buttons:** Use the "📊 Summarize data" or "🎯 Best locations" buttons for instant analysis

---

## ⚙️ Troubleshooting

| Problem | Solution |
|---------|----------|
| "Ollama not running" | Make sure the Ollama app is open in your Mac menu bar |
| "Model not found" | Run `ollama pull tinyllama` again |
| "ModuleNotFoundError: No module named 'ollama'" | Run `pip install ollama` |
| AI taking too long | TinyLlama may be downloading. Check your internet & give it time |
| "Connection refused" | Ollama crashed. Restart the app and `ollama pull tinyllama` |

---

## 📊 How It Works

1. You generate a map with your filters (species, years, months)
2. The AI reads the current data summary
3. You ask questions via chat or quick buttons
4. TinyLlama analyzes patterns and gives insights
5. Everything runs **locally** on your M1 Mac—no internet needed!

---

## 🎯 Example Workflow

1. Select **Yellowfin Tuna** and **July-September**
2. Click **Generate Map**
3. In the sidebar, ask: *"Where do yellowfin tuna migrate in late summer?"*
4. AI gives you instant analysis based on your current filters
5. Change filters → ask new questions → get new insights

---

## 🛑 Stopping the App

- Press `Ctrl+C` in Terminal
- The Ollama app can stay running (it uses minimal RAM when idle)

Enjoy your AI-powered fisheries analysis! 🐟
