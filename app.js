(function () {
  const messagesEl = document.getElementById("messages");
  const chatContainer = document.getElementById("chat-container");
  const userInput = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const clearBtn = document.getElementById("clear-btn");
  const apiHostInput = document.getElementById("api-host");
  const modelSelect = document.getElementById("model-select");
  const refreshModelsBtn = document.getElementById("refresh-models-btn");
  const systemPromptInput = document.getElementById("system-prompt");
  const imageInput = document.getElementById("image-input");
  const previewBar = document.getElementById("preview-bar");
  const reasoningToggle = document.getElementById("reasoning-toggle");
  const speechSpeedInput = document.getElementById("speech-speed");
  const speechSpeedVal = document.getElementById("speech-speed-val");

  function writeString(view, offset, str) {
    for (var i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  }

  function audioBlobToWavDataUrl(blob) {
    return new Promise(function (resolve, reject) {
      var reader = new FileReader();
      reader.onload = function () {
        var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        audioCtx.decodeAudioData(reader.result, function (buffer) {
          var numChannels = buffer.numberOfChannels;
          var sampleRate = buffer.sampleRate;
          var bitsPerSample = 16;
          var samples = buffer.length;
          var dataLength = samples * numChannels * (bitsPerSample / 8);
          var totalLength = 44 + dataLength;
          var wav = new ArrayBuffer(totalLength);
          var view = new DataView(wav);
          writeString(view, 0, "RIFF");
          view.setUint32(4, totalLength - 8, true);
          writeString(view, 8, "WAVE");
          writeString(view, 12, "fmt ");
          view.setUint32(16, 16, true);
          view.setUint16(20, 1, true);
          view.setUint16(22, numChannels, true);
          view.setUint32(24, sampleRate, true);
          view.setUint32(28, sampleRate * numChannels * (bitsPerSample / 8), true);
          view.setUint16(32, numChannels * (bitsPerSample / 8), true);
          view.setUint16(34, bitsPerSample, true);
          writeString(view, 36, "data");
          view.setUint32(40, dataLength, true);
          var channels = [];
          for (var c = 0; c < numChannels; c++) {
            channels.push(buffer.getChannelData(c));
          }
          var offset = 44;
          for (var i = 0; i < samples; i++) {
            for (var ch = 0; ch < numChannels; ch++) {
              var s = Math.max(-1, Math.min(1, channels[ch][i]));
              view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
              offset += 2;
            }
          }
          var wavBlob = new Blob([wav], { type: "audio/wav" });
          var urlReader = new FileReader();
          urlReader.onload = function () { resolve(urlReader.result); };
          urlReader.readAsDataURL(wavBlob);
          audioCtx.close();
        }, reject);
      };
      reader.readAsArrayBuffer(blob);
    });
  }

  let conversation = [];
  let seenIds = new Set();
  let pendingImages = [];
  let pendingAudios = [];
  let abortController = null;
  let lastUserImages = [];
  let isRecording = false;
  let mediaRecorder = null;
  let audioChunks = [];
  let mediaStream = null;

  async function loadModels() {
    var host = apiHostInput.value.replace(/\/+$/, "");
    var prev = modelSelect.value;
    modelSelect.innerHTML = '<option value="">(server default)</option>';
    try {
      var resp = await fetch(host + "/v1/models");
      if (!resp.ok) return;
      var data = await resp.json();
      var models = (data.data || []).sort(function (a, b) {
        return (a.id || "").localeCompare(b.id || "");
      });
      models.forEach(function (m) {
        var opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = m.id;
        modelSelect.appendChild(opt);
      });
      if (prev && Array.from(modelSelect.options).some(function (o) { return o.value === prev; })) {
        modelSelect.value = prev;
      } else if (models.length > 0) {
        modelSelect.value = models[0].id;
      }
    } catch (_) {}
  }

  function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  function autoResize() {
    userInput.style.height = "auto";
    userInput.style.height = Math.min(userInput.scrollHeight, 200) + "px";
  }

  function renderImages(container, images) {
    if (!images || images.length === 0) return;
    const wrap = document.createElement("div");
    wrap.className = "images";
    images.forEach(function (src) {
      const img = document.createElement("img");
      img.src = src;
      wrap.appendChild(img);
    });
    container.appendChild(wrap);
  }

  function renderAudios(container, audios) {
    if (!audios || audios.length === 0) return;
    const wrap = document.createElement("div");
    wrap.className = "audios";
    audios.forEach(function (src) {
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.src = src;
      wrap.appendChild(audio);
    });
    container.appendChild(wrap);
  }

  function addMessage(role, text, images, audios) {
    const el = document.createElement("div");
    el.className = "message " + role;

    const label = document.createElement("div");
    label.className = "role-label";
    label.textContent = role;
    el.appendChild(label);

    if (images && images.length > 0) {
      renderImages(el, images);
    }

    if (audios && audios.length > 0) {
      renderAudios(el, audios);
    }

    const body = document.createElement("div");
    body.className = "message-body";
    body.textContent = text || "";
    el.appendChild(body);

    messagesEl.appendChild(el);
    scrollToBottom();
    return body;
  }

  function addThinking() {
    var el = addMessage("assistant", "");
    el.classList.add("thinking");
    return el;
  }

  function getOrCreateAssistantMessage() {
    let last = messagesEl.lastElementChild;
    if (last && last.classList.contains("assistant")) {
      return last.querySelector(".message-body") || last.querySelector("div:last-child");
    }
    return addMessage("assistant", "");
  }

  function partialThinkTagLength(text) {
    var tags = ["<think>", "</think>"];
    var max = 0;
    tags.forEach(function (tag) {
      for (var i = 1; i < tag.length; i++) {
        if (text.endsWith(tag.slice(0, i))) max = Math.max(max, i);
      }
    });
    return max;
  }

  function splitThinkContent(text) {
    var partial = partialThinkTagLength(text);
    var parseable = partial ? text.slice(0, -partial) : text;
    var answer = "";
    var reasoning = "";
    var inThink = false;
    var i = 0;

    while (i < parseable.length) {
      if (inThink) {
        var end = parseable.indexOf("</think>", i);
        if (end === -1) {
          reasoning += parseable.slice(i);
          break;
        }
        reasoning += parseable.slice(i, end);
        i = end + "</think>".length;
        inThink = false;
      } else {
        var start = parseable.indexOf("<think>", i);
        if (start === -1) {
          answer += parseable.slice(i);
          break;
        }
        answer += parseable.slice(i, start);
        i = start + "<think>".length;
        inThink = true;
      }
    }

    if (reasoning) answer = answer.replace(/^\s+/, "");
    return { answer: answer, reasoning: reasoning };
  }

  function setReasoningText(msgEl, text) {
    var clean = (text || "").trim();
    if (!clean || !msgEl) return;

    var details = msgEl.querySelector(".reasoning");
    if (!details) {
      details = document.createElement("details");
      details.className = "reasoning";

      var summary = document.createElement("summary");
      summary.textContent = "Reasoning";
      details.appendChild(summary);

      var content = document.createElement("div");
      content.className = "reasoning-content";
      details.appendChild(content);

      var body = msgEl.querySelector(".message-body");
      msgEl.insertBefore(details, body || null);
    }

    details.querySelector(".reasoning-content").textContent = clean;
  }

  function renderPreviewBar() {
    previewBar.innerHTML = "";
    pendingImages.forEach(function (src, i) {
      const wrap = document.createElement("div");
      wrap.className = "image-preview";
      const img = document.createElement("img");
      img.src = src;
      const removeBtn = document.createElement("button");
      removeBtn.className = "remove-btn";
      removeBtn.textContent = "×";
      removeBtn.onclick = function () {
        pendingImages.splice(i, 1);
        renderPreviewBar();
      };
      wrap.appendChild(img);
      wrap.appendChild(removeBtn);
      previewBar.appendChild(wrap);
    });
    pendingAudios.forEach(function (src, i) {
      const wrap = document.createElement("div");
      wrap.className = "audio-preview";
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.src = src;
      const removeBtn = document.createElement("button");
      removeBtn.className = "remove-btn";
      removeBtn.textContent = "×";
      removeBtn.onclick = function () {
        pendingAudios.splice(i, 1);
        renderPreviewBar();
      };
      wrap.appendChild(audio);
      wrap.appendChild(removeBtn);
      previewBar.appendChild(wrap);
    });
  }

  function fileToBase64(file) {
    return new Promise(function (resolve) {
      const reader = new FileReader();
      reader.onload = function () {
        resolve(reader.result);
      };
      reader.readAsDataURL(file);
    });
  }

  function buildContent(text, images, audios) {
    var hasImages = images && images.length > 0;
    var hasAudios = audios && audios.length > 0;
    if (!hasImages && !hasAudios) return text || "";
    var parts = [];
    if (hasImages) {
      images.forEach(function (src) {
        parts.push({ type: "image_url", image_url: { url: src } });
      });
    }
    if (hasAudios) {
      audios.forEach(function (src) {
        var match = src.match(/^data:audio\/(\w+);base64,/);
        var format = match ? match[1] : "wav";
        var b64 = src.replace(/^data:audio\/[^;]+;base64,/, "");
        parts.push({ type: "input_audio", input_audio: { data: b64, format: format } });
      });
      return parts;
    }
    parts.push({ type: "text", text: text || "" });
    return parts;
  }

  function buildMessages(text, images, audios) {
    const msgs = [];
    const sys = systemPromptInput.value.trim();
    if (sys) {
      msgs.push({ role: "system", content: sys });
    }
    conversation.forEach(function (m) {
      if (seenIds.has(m.id)) return;
      seenIds.add(m.id);
      msgs.push({ role: m.role, content: m.content });
    });
    msgs.push({ role: "user", content: buildContent(text, images, audios) });
    return msgs;
  }

  var QwenParser = {
    parseResponse: function (text) {
      var bboxes = [];
      var re = /\{[^{}]*"(?:box_2d|bbox_2d)"\s*:\s*\[[^\]]+\][^{}]*\}/g;
      var m;
      while ((m = re.exec(text)) !== null) {
        try {
          var obj = JSON.parse(m[0]);
          if (obj.box_2d) bboxes.push(obj.box_2d);
          else if (obj.bbox_2d) bboxes.push(obj.bbox_2d);
        } catch (_) {}
      }
      return { bboxes: bboxes, divisor: 1000 };
    }
  };

  var GemmaParser = {
    parseResponse: function (text) {
      var bboxes = [];
      var re = /\{[^{}]*"(?:box_2d|bbox_2d)"\s*:\s*\[[^\]]+\][^{}]*\}/g;
      var m;
      while ((m = re.exec(text)) !== null) {
        try {
          var obj = JSON.parse(m[0]);
          var raw = obj.box_2d || obj.bbox_2d;
          if (raw) bboxes.push([raw[1], raw[0], raw[3], raw[2]]);
        } catch (_) {}
      }
      return { bboxes: bboxes, divisor: 1000 };
    }
  };

  var PaligemmaParser = {
    parseResponse: function (text) {
      var bboxes = [];
      var segments = [];

      var segData = null;
      try {
        segData = JSON.parse(text.trim());
      } catch (_) {
        var startIdx = text.indexOf('{"task"');
        if (startIdx !== -1) {
          var depth = 0;
          for (var i = startIdx; i < text.length; i++) {
            if (text[i] === '{') depth++;
            else if (text[i] === '}') {
              depth--;
              if (depth === 0) {
                try { segData = JSON.parse(text.substring(startIdx, i + 1)); } catch (__) {}
                break;
              }
            }
          }
        }
      }

      if (segData && segData.task === "segmentation" && Array.isArray(segData.objects)) {
        segData.objects.forEach(function (obj) {
          if (obj.polygon && Array.isArray(obj.polygon) && obj.polygon.length > 2) {
            segments.push({
              name: obj.name || "",
              bbox: obj.bbox || null,
              polygon: obj.polygon
            });
          }
          if (obj.bbox) bboxes.push(obj.bbox);
        });
        if (segments.length > 0 || bboxes.length > 0) {
          return { bboxes: bboxes, divisor: 1000, segments: segments };
        }
      }

      var re = /<loc(\d+)><loc(\d+)><loc(\d+)><loc(\d+)>/g;
      var m;
      while ((m = re.exec(text)) !== null) {
        var y1 = parseInt(m[1], 10);
        var x1 = parseInt(m[2], 10);
        var y2 = parseInt(m[3], 10);
        var x2 = parseInt(m[4], 10);
        bboxes.push([x1, y1, x2, y2]);
      }
      return { bboxes: bboxes, divisor: 1024, segments: [] };
    }
  };

  function getGroundingParser(modelName) {
    var name = (modelName || "").toLowerCase();
    if (name.includes("paligemma")) return PaligemmaParser;
    if (name.includes("gemma")) return GemmaParser;
    return QwenParser;
  }

  function drawBboxOnImage(imageSrc, bboxes, divisor) {
    return new Promise(function (resolve) {
      var img = new Image();
      img.onload = function () {
        var canvas = document.createElement("canvas");
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        var ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);

        ctx.lineWidth = Math.max(2, Math.round(canvas.width / 250));
        ctx.strokeStyle = "#ff3333";
        ctx.fillStyle = "rgba(255, 50, 50, 0.15)";
        ctx.font = "bold " + Math.max(14, Math.round(canvas.width / 40)) + "px sans-serif";

        for (var i = 0; i < bboxes.length; i++) {
          var bx1 = bboxes[i][0], by1 = bboxes[i][1], bx2 = bboxes[i][2], by2 = bboxes[i][3];
          var x = (bx1 / divisor) * canvas.width;
          var y = (by1 / divisor) * canvas.height;
          var w = ((bx2 - bx1) / divisor) * canvas.width;
          var h = ((by2 - by1) / divisor) * canvas.height;
          ctx.fillRect(x, y, w, h);
          ctx.strokeRect(x, y, w, h);
        }

        resolve(canvas.toDataURL("image/png"));
      };
      img.src = imageSrc;
    });
  }

  function overlayMasksOnImage(imageSrc, maskPngs) {
    return new Promise(function (resolve) {
      var img = new Image();
      img.onload = function () {
        var canvas = document.createElement("canvas");
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        var ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);

        var colors = [
          [255, 50, 50],
          [50, 200, 50],
          [50, 100, 255],
          [255, 200, 50],
          [200, 50, 200],
          [50, 200, 200]
        ];

        var total = maskPngs.length;
        if (total === 0) { resolve(canvas.toDataURL("image/png")); return; }

        var loaded = 0;
        maskPngs.forEach(function (maskB64, idx) {
          var maskImg = new Image();
          maskImg.onload = function () {
            var tempCanvas = document.createElement("canvas");
            tempCanvas.width = canvas.width;
            tempCanvas.height = canvas.height;
            var tempCtx = tempCanvas.getContext("2d");
            tempCtx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);

            var imgData = tempCtx.getImageData(0, 0, canvas.width, canvas.height);
            var pixels = imgData.data;
            var c = colors[idx % colors.length];

            for (var i = 0; i < pixels.length; i += 4) {
              var alpha = pixels[i];
              pixels[i] = c[0];
              pixels[i + 1] = c[1];
              pixels[i + 2] = c[2];
              pixels[i + 3] = Math.round(alpha * 0.4);
            }

            tempCtx.putImageData(imgData, 0, 0);
            ctx.drawImage(tempCanvas, 0, 0);

            loaded++;
            if (loaded === total) {
              resolve(canvas.toDataURL("image/png"));
            }
          };
          maskImg.src = "data:image/png;base64," + maskB64;
        });
      };
      img.src = imageSrc;
    });
  }

  var micBtn = document.getElementById("mic-btn");
  var recordingIndicator = document.getElementById("recording-indicator");

  function setRecordingUI(on) {
    micBtn.classList.toggle("recording", on);
    recordingIndicator.classList.toggle("active", on);
  }

  async function startRecording() {
    if (isRecording || abortController) return;
    try {
      if (!mediaStream || !mediaStream.active) {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      }
    } catch (_) { return; }
    audioChunks = [];
    var mimeType = "audio/webm";
    if (!MediaRecorder.isTypeSupported(mimeType)) {
      mimeType = "audio/ogg";
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = "";
      }
    }
    mediaRecorder = new MediaRecorder(mediaStream, mimeType ? { mimeType: mimeType } : undefined);
    mediaRecorder.ondataavailable = function (e) {
      if (e.data.size > 0) audioChunks.push(e.data);
    };
    mediaRecorder.start();
    isRecording = true;
    setRecordingUI(true);
  }

  function stopRecording() {
    if (!isRecording || !mediaRecorder) return;
    return new Promise(function (resolve) {
      mediaRecorder.onstop = function () {
        var blob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
        audioBlobToWavDataUrl(blob).then(function (dataUrl) {
          pendingAudios.push(dataUrl);
          renderPreviewBar();
          resolve();
        });
        mediaRecorder = null;
        audioChunks = [];
        isRecording = false;
        setRecordingUI(false);
      };
      mediaRecorder.stop();
    });
  }

  var ttsActiveBtn = null;

  async function handleTTS(text, btn) {
    if (!window.KittenTTS) return;
    if (btn.classList.contains('tts-playing')) {
      KittenTTS.stop();
      btn.classList.remove('tts-playing');
      btn.textContent = '\u25b6';
      btn.title = 'Read aloud';
      ttsActiveBtn = null;
      return;
    }
    if (ttsActiveBtn) {
      KittenTTS.stop();
      ttsActiveBtn.classList.remove('tts-playing');
      ttsActiveBtn.textContent = '\u25b6';
      ttsActiveBtn.title = 'Read aloud';
      ttsActiveBtn = null;
    }
    btn.classList.add('tts-loading');
    btn.disabled = true;
    btn.textContent = '\u23f3';
    try {
      if (!KittenTTS.isReady()) {
        await KittenTTS.init(function (m) { btn.title = m || 'Loading TTS...'; });
      }
      btn.classList.remove('tts-loading');
      btn.classList.add('tts-playing');
      btn.textContent = '\u25a0';
      btn.title = 'Stop';
      btn.disabled = false;
      ttsActiveBtn = btn;
      await KittenTTS.speak(text);
    } catch (e) {
      btn.title = 'TTS error: ' + e.message;
    } finally {
      btn.classList.remove('tts-loading', 'tts-playing');
      btn.textContent = '\u25b6';
      btn.title = 'Read aloud';
      btn.disabled = false;
      if (ttsActiveBtn === btn) ttsActiveBtn = null;
    }
  }

  function addTTSButton(msgEl, text) {
    if (!window.KittenTTS || !text) return;
    var btn = document.createElement('button');
    btn.className = 'tts-btn';
    btn.textContent = '\u25b6';
    btn.title = 'Read aloud';
    (function (t, b) {
      b.addEventListener('click', function () { handleTTS(t, b); });
    })(text, btn);
    msgEl.appendChild(btn);
  }

  async function send() {
    const text = userInput.value.trim();
    if (!text && pendingImages.length === 0 && pendingAudios.length === 0) return;
    if (abortController) return;

    const images = pendingImages.slice();
    const audios = pendingAudios.slice();
    pendingImages = [];
    pendingAudios = [];
    renderPreviewBar();

    addMessage("user", text, images, audios);
    userInput.value = "";
    autoResize();

    lastUserImages = images;

    abortController = new AbortController();
    sendBtn.disabled = true;

    const host = apiHostInput.value.replace(/\/+$/, "");
    const body = {
      messages: buildMessages(text, images, audios),
      stream: true,
      stream_options: { include_usage: true },
    };
    
    conversation.push({ id: "user-" + Date.now(), role: "user", content: buildContent(text, images, audios) });
    const model = modelSelect.value;
    if (model) body.model = model;
    if (reasoningToggle.checked) {
      body.chat_template_kwargs = { enable_thinking: true };
    }

      let rawContentText = "";
      let responseText = "";
      let inlineReasoningText = "";
      let reasoningContentText = "";
      let aborted = false;
      let collectedMaskPngs = [];

    const bodyEl = addThinking();
    const requestStartTime = performance.now();

    try {
      const resp = await fetch(host + "/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: abortController.signal,
      });

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let gotContent = false;
      let tokenCount = 0;
      let usage = null;
      let firstTokenTime = 0;
      let lastTokenTime = 0;

      while (true) {
        const chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith("data: ")) continue;
          const data = trimmed.slice(6);
          if (data === "[DONE]") break;
          try {
            const json = JSON.parse(data);
            if (json.usage) usage = json.usage;
            const choice = json.choices && json.choices[0];
            const delta = choice && choice.delta;
            if (choice && choice.mask_pngs && choice.mask_pngs.length > 0) {
              collectedMaskPngs = collectedMaskPngs.concat(choice.mask_pngs);
            }
            if (delta && (delta.reasoning_content || delta.reasoning)) {
              var reasoningNow = performance.now();
              if (!firstTokenTime) firstTokenTime = reasoningNow;
              tokenCount++;
              lastTokenTime = reasoningNow;

              var deltaReasoning = delta.reasoning_content || delta.reasoning;
              reasoningContentText += typeof deltaReasoning === "string" ? deltaReasoning : JSON.stringify(deltaReasoning);
              setReasoningText(bodyEl.parentElement, reasoningContentText + inlineReasoningText);
              scrollToBottom();
            }
            if (delta && delta.content) {
              var now = performance.now();
              if (!firstTokenTime) firstTokenTime = now;
              tokenCount++;
              lastTokenTime = now;

              rawContentText += delta.content;
              var parsed = splitThinkContent(rawContentText);
              inlineReasoningText = parsed.reasoning;
              setReasoningText(bodyEl.parentElement, reasoningContentText + inlineReasoningText);

              if (parsed.answer && !gotContent) {
                gotContent = true;
                bodyEl.classList.remove("thinking");
              }
              responseText = parsed.answer;
              bodyEl.textContent = responseText;
              scrollToBottom();
            }
          } catch (_) {}
        }
      }

      var completionTokens = usage && typeof usage.completion_tokens === "number" ? usage.completion_tokens : tokenCount;
      if (completionTokens > 0 && firstTokenTime) {
        var ttft = (firstTokenTime - requestStartTime) / 1000;
        var genTokens = Math.max(completionTokens - 1, 0);
        var genElapsed = (lastTokenTime - firstTokenTime) / 1000;
        var tps = genTokens > 0 && genElapsed > 0 ? (genTokens / genElapsed).toFixed(1) : "—";
        var statsEl = document.createElement("div");
        statsEl.className = "gen-stats";
        statsEl.textContent = "TTFT " + ttft.toFixed(2) + "s \u00b7 gen " + tps + " tok/s \u00b7 " + completionTokens + " tokens";
        bodyEl.parentElement.appendChild(statsEl);
      }

      if (!responseText && !resp.ok) {
        responseText = "Error " + resp.status + ": " + buffer;
        bodyEl.textContent = responseText;
      }
    } catch (e) {
      if (e.name !== "AbortError") {
        responseText = "Error: " + e.message;
        getOrCreateAssistantMessage().textContent = responseText;
      } else {
        aborted = true;
      }
    } finally {
      if (!responseText && !aborted) {
        responseText = "(empty response)";
        bodyEl.classList.remove("thinking");
        bodyEl.textContent = responseText;
      }

      if (responseText && !conversation[conversation.length - 1] || conversation[conversation.length - 1].role !== "assistant") {
        conversation.push({ id: "assistant-" + Date.now(), role: "assistant", content: responseText });
      }

      if (responseText) {
        var ttsMsg = messagesEl.lastElementChild;
        if (ttsMsg && ttsMsg.classList.contains('assistant')) {
          addTTSButton(ttsMsg, responseText);
        }

        var parser = getGroundingParser(model || modelSelect.value);
        var result = parser.parseResponse(responseText);
        if (lastUserImages.length > 0 && (collectedMaskPngs.length > 0 || result.bboxes.length > 0)) {
          try {
            var annotatedSrc;
            if (collectedMaskPngs.length > 0) {
              annotatedSrc = await overlayMasksOnImage(lastUserImages[0], collectedMaskPngs);
            } else {
              annotatedSrc = await drawBboxOnImage(lastUserImages[0], result.bboxes, result.divisor);
            }
            var msgEl = messagesEl.lastElementChild;
            if (msgEl) {
              var wrap = document.createElement("div");
              wrap.className = "bbox-image";
              var aimg = document.createElement("img");
              aimg.src = annotatedSrc;
              wrap.appendChild(aimg);
              msgEl.appendChild(wrap);
              scrollToBottom();
            }
           } catch (_) {}
         }
       }

       abortController = null;
       sendBtn.disabled = false;
       userInput.focus();
     }
   }

   sendBtn.addEventListener("click", send);

  userInput.addEventListener("keydown", function (e) {
    if ((e.key === "Enter" && !e.shiftKey) || (e.key === "Enter" && (e.metaKey || e.ctrlKey))) {
      e.preventDefault();
      send();
    }
  });

  userInput.addEventListener("input", autoResize);

  refreshModelsBtn.addEventListener("click", loadModels);

  speechSpeedInput.addEventListener("input", function () {
    var val = parseFloat(speechSpeedInput.value);
    speechSpeedVal.textContent = val.toFixed(1) + "x";
    if (window.KittenTTS) KittenTTS.setSpeed(val);
  });

  imageInput.addEventListener("change", async function () {
    const files = Array.from(imageInput.files);
    for (const f of files) {
      const b64 = await fileToBase64(f);
      pendingImages.push(b64);
    }
    imageInput.value = "";
    renderPreviewBar();
  });

  clearBtn.addEventListener("click", function () {
    conversation = [];
    seenIds = new Set();
    pendingImages = [];
    pendingAudios = [];
    lastUserImages = [];
    renderPreviewBar();
    messagesEl.innerHTML = "";
  });

  document.addEventListener("keydown", function (e) {
    if (e.code === "Space" && e.shiftKey && document.activeElement !== userInput && !e.repeat) {
      e.preventDefault();
      startRecording();
    }
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && document.activeElement !== userInput) {
      e.preventDefault();
      send();
    }
  });

  document.addEventListener("keyup", function (e) {
    if (e.code === "Space" && isRecording) {
      e.preventDefault();
      stopRecording();
    }
  });

  micBtn.addEventListener("mousedown", function (e) {
    e.preventDefault();
    startRecording();
  });

  document.addEventListener("mouseup", function () {
    if (isRecording) stopRecording();
  });

  var lightbox = document.getElementById("lightbox");
  var lightboxImg = document.getElementById("lightbox-img");

  function openLightbox(src) {
    lightboxImg.src = src;
    lightbox.classList.add("active");
  }

  function closeLightbox() {
    lightbox.classList.remove("active");
    lightboxImg.src = "";
  }

  lightbox.addEventListener("click", function () {
    closeLightbox();
  });

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && lightbox.classList.contains("active")) {
      closeLightbox();
    }
  });

  messagesEl.addEventListener("click", function (e) {
    if (e.target.tagName === "IMG" && e.target.closest(".message")) {
      openLightbox(e.target.src);
    }
  });

  loadModels();
})();
