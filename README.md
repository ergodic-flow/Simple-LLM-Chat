# LLM Chat UI

A simple multi-modal chat UI. Accepts images, audio using the microphone, and text inputs.
Supports simple TTS via KittenTTS in browser.

Works with any OpenAI compatible API server, including VLLM and llama-server.

Just run your server, and point the UI to the server URL. No dependencies needed, just open `index.html` in a browser.

## Features

Hold to speak to model via microphone input.
Image inputs.
Draws bounding boxes for supported models such as Gemma and Qwen.
TTS of LLM outputs using KittenTTS. Takes a minute to download models, but then fast.
