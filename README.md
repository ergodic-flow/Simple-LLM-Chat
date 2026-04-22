# Simple LLM Chat

A simple multi-modal chat UI. Accepts images, audio, and text inputs.

Talk to your LLM, text your llm, or show it pictures. TTS supported via KittenTTS in browser.

Just run your server, and point the UI to the server URL. No installation, just open `index.html` in a browser.

## Features

- Works with any OpenAI compatible API server, including VLLM and llama-server.
- Hold to speak to model via microphone input.
- Image inputs.
- Draws bounding boxes for supported models such as Gemma and Qwen.
- TTS of LLM outputs using KittenTTS. Takes a minute to download models, but then fast.
