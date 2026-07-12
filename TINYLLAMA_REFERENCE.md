# TinyLlama on M1 MacBook Air - Reference Guide

## About TinyLlama
- **Size**: 1.1GB (after download)
- **RAM Usage**: ~2-3GB when running (perfect for 8GB)
- **Speed on M1**: Very fast (hardware optimized)
- **Quality**: Good for analysis & Q&A tasks
- **Cost**: Free & Open Source
- **Offline**: Yes, completely local

## Installation Check

Verify everything is set up correctly:

```bash
# Check if Ollama is installed
which ollama

# Check if TinyLlama is downloaded
ollama list

# Test the connection
curl http://localhost:11434/api/tags
```

## Ollama Commands You Might Need

```bash
# Start Ollama (should auto-start on Mac)
ollama serve

# Download a model
ollama pull tinyllama

# List downloaded models
ollama list

# Remove a model (if you need space)
ollama rm tinyllama

# Test the model from command line
ollama run tinyllama "Explain fish migration patterns"
```

## M1 Optimization Notes

- Ollama auto-detects M1 and uses GPU acceleration
- No special setup needed—it just works
- The app runs faster than on Intel Macs
- RAM usage stays low due to M1's efficiency

## Changing Models (Advanced)

If you want to experiment with other tiny models:

```bash
# Neural Chat (slightly smarter, 4.1GB)
ollama pull neural-chat

# Phi 2 (even smaller, 2.7GB)
ollama pull phi
```

Then edit `fish_app.py`, find the `MODEL_NAME = 'tinyllama'` line near the top
configuration section and change it:
```python
MODEL_NAME = 'neural-chat'  # or 'phi'
```

## Common Issues & Fixes

**Issue**: "Ollama command not found"
```bash
# Add Ollama to PATH
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**Issue**: Port 11434 already in use
```bash
# Find what's using it
lsof -i :11434

# Kill it (replace XXXX with PID)
kill -9 XXXX
```

**Issue**: Model downloads too slowly
- Check internet speed
- Try downloading again: `ollama pull tinyllama`
- Models are cached, so it won't re-download

**Issue**: AI responses are slow
- First response is slower (model warming up)
- Subsequent responses are faster
- This is normal for local models

## Resources

- Ollama docs: https://ollama.ai
- TinyLlama GitHub: https://github.com/jzhang38/TinyLlama
- Model options: https://ollama.ai/library

---

**Pro Tip**: You can also use Ollama from the command line without the app:
```bash
ollama run tinyllama "Your question here"
```

This is useful for testing or debugging!
