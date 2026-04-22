(function () {
  var REPO = 'onnx-community/KittenTTS-Nano-v0.8-ONNX';
  var HF = 'https://huggingface.co';
  var SR = 24000;
  var IDB = 'kittentts';
  var IDB_V = 1;

  var session, voices, config, phonemizeFn, ort, audioCtx, curSrc, stopCb;
  var state = 'idle';
  var initP = null;

  // ========== Tokenizer ==========
  // Symbol table must match KittenTTS Python TextCleaner exactly
  // Source: https://github.com/KittenML/KittenTTS
  var _pad = '$';
  var _punc = ';:,.!?\u00a1\u00bf\u2014\u2026\u201c\u00ab\u00bb\u201d\" ';
  var _let = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
  var _ipa = 'ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘\u0027\u0329\u0027\u1d3b';
  var syms = [_pad];
  for (var i = 0; i < _punc.length; i++) syms.push(_punc[i]);
  for (var i = 0; i < _let.length; i++) syms.push(_let[i]);
  for (var i = 0; i < _ipa.length; i++) syms.push(_ipa[i]);
  var c2i = {};
  for (var i = 0; i < syms.length; i++) c2i[syms[i]] = i;

  function tokenize(phonemes) {
    var ids = [];
    for (var i = 0; i < phonemes.length; i++) {
      var idx = c2i[phonemes[i]];
      if (idx !== undefined) ids.push(idx);
    }
    ids.unshift(0);
    ids.push(10);
    ids.push(0);
    return ids;
  }

  // ========== NPZ / Voice Loader ==========
  function parseNpyHeader(bytes) {
    if (bytes[0] !== 0x93) throw new Error('Not a .npy file');
    var hdr = String.fromCharCode(bytes[1], bytes[2], bytes[3], bytes[4], bytes[5]);
    if (hdr !== 'NUMPY') throw new Error('Not a .npy file');
    var view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    var ver = bytes[6];
    var hLen, hOff;
    if (ver === 1) { hLen = view.getUint16(8, true); hOff = 10; }
    else { hLen = view.getUint32(8, true); hOff = 12; }
    var hStr = new TextDecoder().decode(bytes.slice(hOff, hOff + hLen));
    var dM = hStr.match(/'descr'\s*:\s*'([^']+)'/);
    var sM = hStr.match(/'shape'\s*:\s*\(([^)]*)\)/);
    if (!dM) throw new Error('Bad npy header');
    var shape = sM ? sM[1].split(',').map(function (s) { var n = parseInt(s.trim(), 10); return isNaN(n) ? 0 : n; }).filter(function (n) { return n > 0; }) : [];
    return { descr: dM[1], shape: shape, dataOffset: hOff + hLen };
  }

  function npyToFloat32(bytes) {
    var p = parseNpyHeader(bytes);
    var raw = bytes.slice(p.dataOffset);
    var aligned = new ArrayBuffer(raw.length);
    new Uint8Array(aligned).set(raw);
    var data;
    if (p.descr === '<f4' || p.descr === 'float32') {
      data = new Float32Array(aligned);
    } else if (p.descr === '<f8' || p.descr === 'float64') {
      var f64 = new Float64Array(aligned);
      data = new Float32Array(f64.length);
      for (var i = 0; i < f64.length; i++) data[i] = f64[i];
    } else {
      throw new Error('Unsupported dtype: ' + p.descr);
    }
    return { data: data, shape: p.shape };
  }

  async function extractZip(buf) {
    var bytes = new Uint8Array(buf);
    var view = new DataView(buf);
    var entries = new Map();
    var eocd = -1;
    for (var i = bytes.length - 22; i >= 0; i--) {
      if (view.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
    }
    if (eocd === -1) throw new Error('Not a zip file');
    var cdOff = view.getUint32(eocd + 16, true);
    var cdN = view.getUint16(eocd + 10, true);
    var cdPos = cdOff;
    var cdList = [];
    for (var ci = 0; ci < cdN; ci++) {
      if (view.getUint32(cdPos, true) !== 0x02014b50) break;
      var cm = view.getUint16(cdPos + 10, true);
      var cs = view.getUint32(cdPos + 20, true);
      var us = view.getUint32(cdPos + 24, true);
      var fnl = view.getUint16(cdPos + 28, true);
      var el = view.getUint16(cdPos + 30, true);
      var cl = view.getUint16(cdPos + 32, true);
      var lho = view.getUint32(cdPos + 42, true);
      var fn = new TextDecoder().decode(bytes.slice(cdPos + 46, cdPos + 46 + fnl));
      cdList.push({ fn: fn, cs: cs, us: us, lho: lho, cm: cm });
      cdPos += 46 + fnl + el + cl;
    }
    for (var ci = 0; ci < cdList.length; ci++) {
      var cd = cdList[ci];
      var lhFnl = view.getUint16(cd.lho + 26, true);
      var lhEl = view.getUint16(cd.lho + 28, true);
      var dStart = cd.lho + 30 + lhFnl + lhEl;
      var fd;
      if (cd.cm === 0) {
        fd = bytes.slice(dStart, dStart + cd.us);
      } else if (cd.cm === 8) {
        var comp = bytes.slice(dStart, dStart + cd.cs);
        var ds = new DecompressionStream('deflate-raw');
        var w = ds.writable.getWriter();
        w.write(comp);
        w.close();
        var r = ds.readable.getReader();
        var chunks = [];
        var tl = 0;
        while (true) {
          var rr = await r.read();
          if (rr.done) break;
          chunks.push(rr.value);
          tl += rr.value.length;
        }
        fd = new Uint8Array(tl);
        var pos = 0;
        for (var j = 0; j < chunks.length; j++) { fd.set(chunks[j], pos); pos += chunks[j].length; }
      } else {
        continue;
      }
      entries.set(cd.fn, fd);
    }
    return entries;
  }

  async function parseVoices(buf) {
    var entries = await extractZip(buf);
    var out = {};
    entries.forEach(function (data, name) {
      if (!name.endsWith('.npy')) return;
      var vname = name.replace(/\.npy$/, '');
      var r = npyToFloat32(data);
      out[vname] = { data: r.data, shape: [r.shape[0] || 1, r.shape[1] || r.data.length] };
    });
    return out;
  }

  // ========== IndexedDB Cache ==========
  function idbOpen() {
    return new Promise(function (ok, fail) {
      var r = indexedDB.open(IDB, IDB_V);
      r.onupgradeneeded = function (e) {
        var db = e.target.result;
        if (!db.objectStoreNames.contains('c')) db.createObjectStore('c');
      };
      r.onsuccess = function (e) { ok(e.target.result); };
      r.onerror = function () { fail(null); };
    });
  }

  function idbGet(key) {
    return idbOpen().then(function (db) {
      return new Promise(function (ok) {
        try {
          var tx = db.transaction('c', 'readonly');
          var r = tx.objectStore('c').get(key);
          r.onsuccess = function () { ok(r.result || null); };
          r.onerror = function () { ok(null); };
        } catch (e) { ok(null); }
      });
    }).catch(function () { return null; });
  }

  function idbPut(key, val) {
    return idbOpen().then(function (db) {
      return new Promise(function (ok) {
        try {
          var tx = db.transaction('c', 'readwrite');
          tx.objectStore('c').put(val, key);
          tx.oncomplete = function () { ok(); };
          tx.onerror = function () { ok(); };
        } catch (e) { ok(); }
      });
    }).catch(function () {});
  }

  // ========== Dynamic Dependency Loading ==========
  function ensureOrt() {
    if (ort) return Promise.resolve();
    if (window.ort) { ort = window.ort; return Promise.resolve(); }
    return new Promise(function (ok, fail) {
      var s = document.createElement('script');
      s.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.all.min.js';
      s.onload = function () { ort = window.ort; ok(); };
      s.onerror = function () { fail(new Error('Failed to load ONNX Runtime')); };
      document.head.appendChild(s);
    });
  }

  function ensurePhonemizer() {
    if (phonemizeFn) return Promise.resolve();
    return import('https://esm.sh/phonemizer@1.2.1').then(function (mod) {
      phonemizeFn = mod.phonemize;
    });
  }

  // ========== Network Helpers ==========
  function resolveUrl(file) {
    return HF + '/' + REPO + '/resolve/main/' + file;
  }

  function fetchBuf(url) {
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error('Fetch failed: ' + r.status);
      return r.arrayBuffer();
    });
  }

  async function fetchWithProgress(url, onPct) {
    var resp = await fetch(url);
    if (!resp.ok) throw new Error('Fetch failed: ' + resp.status);
    var total = parseInt(resp.headers.get('content-length') || '0', 10);
    var reader = resp.body.getReader();
    var chunks = [];
    var loaded = 0;
    while (true) {
      var r = await reader.read();
      if (r.done) break;
      chunks.push(r.value);
      loaded += r.value.length;
      if (total > 0) onPct(Math.round(loaded / total * 100));
    }
    var buf = new Uint8Array(loaded);
    var off = 0;
    for (var i = 0; i < chunks.length; i++) { buf.set(chunks[i], off); off += chunks[i].length; }
    return buf.buffer;
  }

  // ========== Model Init ==========
  async function doInit(onProgress) {
    state = 'loading';
    var msg = function (m) { if (onProgress) onProgress(m); };

    msg('Loading runtimes...');
    await Promise.all([ensureOrt(), ensurePhonemizer()]);

    var cached = await Promise.all([idbGet('cfg'), idbGet('mdl'), idbGet('vcs')]);
    var cfgStr = cached[0], mdlBuf = cached[1], vcsBuf = cached[2];

    if (cfgStr && mdlBuf && vcsBuf) {
      msg('Loading cached model...');
      config = JSON.parse(cfgStr);
      voices = await parseVoices(vcsBuf);
      msg('Initializing session...');
      session = await createSession(mdlBuf);
      state = 'ready';
      return;
    }

    msg('Fetching config...');
    var cfgResp = await fetch(resolveUrl('kitten_config.json'));
    if (!cfgResp.ok) cfgResp = await fetch(resolveUrl('config.json'));
    if (!cfgResp.ok) throw new Error('Failed to fetch config');
    config = await cfgResp.json();
    cfgStr = JSON.stringify(config);

    msg('Downloading model...');
    mdlBuf = await fetchWithProgress(resolveUrl('onnx/model.onnx'), function (pct) {
      msg('Downloading model... ' + pct + '%');
    });

    msg('Downloading voices...');
    vcsBuf = await fetchBuf(resolveUrl(config.voices));

    msg('Loading voices...');
    voices = await parseVoices(vcsBuf);

    msg('Initializing session...');
    session = await createSession(mdlBuf);

    msg('Caching for offline use...');
    await Promise.all([idbPut('cfg', cfgStr), idbPut('mdl', mdlBuf), idbPut('vcs', vcsBuf)]);

    state = 'ready';
  }

  async function createSession(mdlBuf) {
    var gpu = false;
    try {
      if ('gpu' in navigator) {
        var adapter = await navigator.gpu.requestAdapter();
        gpu = !!adapter;
      }
    } catch (e) {}

    var opts = {};
    if (gpu) {
      opts.executionProviders = ['webgpu'];
    } else {
      opts.executionProviders = ['wasm'];
      ort.env.wasm.numThreads = 1;
    }

    return ort.InferenceSession.create(mdlBuf, opts);
  }

  // ========== Text Processing ==========
  function ensurePunct(s) {
    s = s.trim();
    if (s && '.!?,;:'.indexOf(s[s.length - 1]) === -1) s += '.';
    return s;
  }

  function chunkText(text) {
    text = text.trim();
    if (!text) return [];
    var sentences = text.match(/[^.!?]*[.!?]+|[^.!?]+$/g) || [text];
    var chunks = [];
    for (var i = 0; i < sentences.length; i++) {
      var s = sentences[i].trim();
      if (!s) continue;
      if (s.length <= 400) {
        chunks.push(ensurePunct(s));
      } else {
        var words = s.split(/\s+/);
        var chunk = '';
        for (var j = 0; j < words.length; j++) {
          if (chunk.length + words[j].length + 1 <= 400) {
            chunk += (chunk ? ' ' : '') + words[j];
          } else {
            if (chunk) chunks.push(ensurePunct(chunk));
            chunk = words[j];
          }
        }
        if (chunk) chunks.push(ensurePunct(chunk));
      }
    }
    return chunks;
  }

  var PUNCT_RE = /(\s*[;:,.!?\u00a1\u00bf\u2014\u2026\u201c\u201d\u00ab\u00bb"()\[\]{}]+\s*)+/g;

  async function phonemizeChunk(text) {
    var sections = [];
    var lastIdx = 0;
    var m;
    PUNCT_RE.lastIndex = 0;
    while ((m = PUNCT_RE.exec(text)) !== null) {
      if (lastIdx < m.index) {
        sections.push({ p: false, t: text.slice(lastIdx, m.index) });
      }
      sections.push({ p: true, t: m[0] });
      lastIdx = m.index + m[0].length;
    }
    if (lastIdx < text.length) {
      sections.push({ p: false, t: text.slice(lastIdx) });
    }
    var parts = await Promise.all(sections.map(function (s) {
      if (s.p) return Promise.resolve(s.t);
      return phonemizeFn(s.t, 'en-us').then(function (r) { return r.join(' '); });
    }));
    var raw = parts.join('');
    var toks = raw.match(/[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s]/gu) || [];
    return toks.join(' ');
  }

  // ========== Synthesis ==========
  function defaultVoiceKey() {
    if (config.voice_aliases) return Object.keys(config.voice_aliases)[0];
    return Object.keys(voices)[0];
  }

  function resolveVoice(key) {
    if (config.voice_aliases && config.voice_aliases[key]) return config.voice_aliases[key];
    return key;
  }

  var speedMult = 1.0;

  async function synthesizeChunk(text) {
    var vk = defaultVoiceKey();
    var vid = resolveVoice(vk);
    var vd = voices[vid];
    if (!vd) throw new Error('Voice not found: ' + vid);

    var speed = speedMult;
    if (config.speed_priors && config.speed_priors[vid]) speed *= config.speed_priors[vid];

    var phonemes = await phonemizeChunk(text);
    var ids = tokenize(phonemes);

    var refId = Math.min(text.length, vd.shape[0] - 1);
    var sDim = vd.shape[1];
    var refStyle = vd.data.slice(refId * sDim, (refId + 1) * sDim);

    var idsT = new ort.Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
    var styleT = new ort.Tensor('float32', refStyle, [1, sDim]);
    var speedT = new ort.Tensor('float32', new Float32Array([speed]), [1]);

    var results = await session.run({ input_ids: idsT, style: styleT, speed: speedT });
    var audio = results[session.outputNames[0]].data;

    if (audio.length > SR) audio = audio.slice(0, audio.length - 5000);
    return new Float32Array(audio);
  }

  async function synthesize(text) {
    var chunks = chunkText(text);
    var parts = [];
    for (var i = 0; i < chunks.length; i++) {
      var a = await synthesizeChunk(chunks[i]);
      parts.push(a);
    }
    var total = 0;
    for (var i = 0; i < parts.length; i++) total += parts[i].length;
    var out = new Float32Array(total);
    var off = 0;
    for (var i = 0; i < parts.length; i++) { out.set(parts[i], off); off += parts[i].length; }
    return out;
  }

  // ========== Audio Playback ==========
  function playFloat32(samples) {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    stopAudio();
    if (audioCtx.state === 'suspended') audioCtx.resume();
    var buf = audioCtx.createBuffer(1, samples.length, SR);
    buf.getChannelData(0).set(samples);
    curSrc = audioCtx.createBufferSource();
    curSrc.buffer = buf;
    curSrc.connect(audioCtx.destination);
    curSrc.start(0);
    return new Promise(function (resolve) {
      curSrc.onended = function () { curSrc = null; resolve(); };
      stopCb = resolve;
    });
  }

  function stopAudio() {
    if (curSrc) { try { curSrc.stop(); } catch (e) {} curSrc = null; }
    if (stopCb) { var cb = stopCb; stopCb = null; cb(); }
  }

  // ========== Public API ==========
  window.KittenTTS = {
    init: function (onProgress) {
      if (state === 'ready') return Promise.resolve();
      if (initP) return initP;
      initP = doInit(onProgress).catch(function (e) {
        state = 'idle';
        initP = null;
        throw e;
      });
      return initP;
    },
    speak: function (text) {
      return synthesize(text).then(function (audio) {
        return playFloat32(audio);
      });
    },
    stop: function () {
      stopAudio();
    },
    isReady: function () {
      return state === 'ready';
    },
    getState: function () {
      return state;
    },
    setSpeed: function (s) {
      speedMult = s;
    }
  };
})();
