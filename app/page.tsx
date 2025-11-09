// app/page.tsx
"use client";

import { useRef, useState, useEffect, useCallback } from "react";
import Tesseract from "tesseract.js";

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const textAreaRef = useRef<HTMLTextAreaElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [status, setStatus] = useState('Click "Start Camera" to begin.');
  const [extractedText, setExtractedText] = useState('Extracted text will appear here...');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isAutoScan, setIsAutoScan] = useState(false);
  const [isBlindMode, setIsBlindMode] = useState(false);
  const [confidence, setConfidence] = useState(0);
  const rafRef = useRef<number | null>(null);
  const previousTextRef = useRef(''); // For stability detection
  const scanCountRef = useRef(0); // Count stable scans
  const accumulatedTextRef = useRef(''); // Accumulate for full page

  // ---- NEW: language + voice state ----
  type Lang = "en" | "hi";
  const [selectedLanguage, setSelectedLanguage] = useState<"en" | "hi" | "both">("en");
  const voiceEn = useRef<SpeechSynthesisVoice | null>(null);
  const voiceHi = useRef<SpeechSynthesisVoice | null>(null);
  const [hasHindiVoice, setHasHindiVoice] = useState(false);

  // speech queue
  const speechQueue = useRef<{ text: string; lang: Lang }[]>([]);
  const isSpeaking = useRef(false);
  const lastAnnounceTime = useRef(0);
  const lastSpokenText = useRef("");

  // === Your Tesseract config (unchanged) ===
  const tesseractConfig = {
    logger: (m: any) => console.log(m),
    oem: 1 as const,
    psm: 6 as const,
    tessedit_char_blacklist: '|{}()[]' as const,
    tessedit_create_pdf: '0' as const,
  };

  // === chunkText (unchanged) ===
  const chunkText = (text: string, maxWords: number = 200): string[] => {
    const words = text.split(/\s+/);
    const chunks: string[] = [];
    let currentChunk = '';
    for (const word of words) {
      if ((currentChunk.split(/\s+/).length + 1) > maxWords) {
        if (currentChunk) chunks.push(currentChunk.trim());
        currentChunk = word + ' ';
      } else {
        currentChunk += word + ' ';
      }
    }
    if (currentChunk) chunks.push(currentChunk.trim());
    return chunks;
  };

  // === Preexisting image processing helpers (unchanged) ===
  const calculateOtsuThreshold = (data: Uint8ClampedArray, width: number, height: number): number => {
    const histogram = new Array(256).fill(0);
    for (let i = 0; i < data.length; i += 4) {
      histogram[data[i]]++;
    }

    let total = width * height;
    let sum = 0;
    for (let i = 0; i < 256; i++) sum += i * histogram[i];

    let sumB = 0;
    let wB = 0;
    let wF = 0;
    let max = 0;
    let threshold = 0;

    for (let t = 0; t < 256; t++) {
      wB += histogram[t];
      if (wB === 0) continue;
      wF = total - wB;
      if (wF === 0) break;
      sumB += t * histogram[t];
      const mB = sumB / wB;
      const mF = (sum - sumB) / wF;
      const between = wB * wF * (mB - mF) * (mB - mF);
      if (between > max) {
        max = between;
        threshold = t;
      }
    }
    return threshold;
  };

  const preprocessImage = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, video: HTMLVideoElement) => {
    const targetHeight = 1200;
    let { videoWidth, videoHeight } = video;
    const aspectRatio = videoWidth / videoHeight;

    if (videoHeight > targetHeight) {
      videoHeight = targetHeight;
      videoWidth = videoHeight * aspectRatio;
    } else if (videoHeight < 600) {
      const scale = 600 / videoHeight;
      videoHeight *= scale;
      videoWidth *= scale;
    }

    canvas.width = videoWidth;
    canvas.height = videoHeight;
    ctx.drawImage(video, 0, 0, videoWidth, videoHeight);

    // Grayscale
    let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
      const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
      data[i] = data[i + 1] = data[i + 2] = avg;
    }
    ctx.putImageData(imageData, 0, 0);

    // Median filter (simple 3x3)
    const medianFilter = () => {
      const tempData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const temp = tempData.data;
      for (let y = 1; y < canvas.height - 1; y++) {
        for (let x = 1; x < canvas.width - 1; x++) {
          const idx = (y * canvas.width + x) * 4;
          const neighbors = [
            temp[(y - 1) * canvas.width * 4 + (x - 1) * 4],
            temp[(y - 1) * canvas.width * 4 + x * 4],
            temp[(y - 1) * canvas.width * 4 + (x + 1) * 4],
            temp[idx - canvas.width * 4],
            temp[idx],
            temp[idx + canvas.width * 4],
            temp[(y + 1) * canvas.width * 4 + (x - 1) * 4],
            temp[(y + 1) * canvas.width * 4 + x * 4],
            temp[(y + 1) * canvas.width * 4 + (x + 1) * 4],
          ].sort((a, b) => a - b);
          const median = neighbors[4];
          data[idx] = data[idx + 1] = data[idx + 2] = median;
        }
      }
      ctx.putImageData(tempData, 0, 0);
    };
    medianFilter();

    // Otsu binarization
    imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    data = imageData.data;
    const threshold = calculateOtsuThreshold(data, canvas.width, canvas.height);
    for (let i = 0; i < data.length; i += 4) {
      const pixel = data[i];
      data[i] = data[i + 1] = data[i + 2] = pixel > threshold ? 255 : 0;
    }
    ctx.putImageData(imageData, 0, 0);

    // Sharpen/contrast
    ctx.filter = "contrast(1.2) brightness(1.1)";
    ctx.drawImage(canvas, 0, 0);
    ctx.filter = "none";

    ctx.strokeStyle = "white";
    ctx.lineWidth = 3;
    ctx.strokeRect(1, 1, canvas.width - 2, canvas.height - 2);

    return canvas.toDataURL("image/png");
  };

  // ---- NEW: speech system (Hindi + English) ----

  // pick best voices available on client
  const loadVoices = () => {
    if (!("speechSynthesis" in window)) return;
    const voices = speechSynthesis.getVoices() || [];

    voiceEn.current =
      voices.find((v) => v.lang?.toLowerCase().startsWith("en") && v.name?.toLowerCase().includes("google")) ||
      voices.find((v) => v.lang?.toLowerCase().startsWith("en")) ||
      null;

    voiceHi.current =
      voices.find((v) => v.lang?.toLowerCase().startsWith("hi")) ||
      voices.find((v) => v.name?.toLowerCase().includes("hindi")) ||
      null;

    setHasHindiVoice(!!voiceHi.current);
    console.log("voiceEn", voiceEn.current?.name, voiceEn.current?.lang);
    console.log("voiceHi", voiceHi.current?.name, voiceHi.current?.lang);
  };

  useEffect(() => {
    // load voices; browsers may return [] initially so also attach onvoiceschanged
    loadVoices();
    if ("speechSynthesis" in window) speechSynthesis.onvoiceschanged = loadVoices;
  }, []);

  const processSpeechQueue = () => {
    if (isSpeaking.current) return;
    const next = speechQueue.current.shift();
    if (!next) return;
    isSpeaking.current = true;
    lastSpokenText.current = next.text;

    const utter = new SpeechSynthesisUtterance(next.text);
    utter.rate = next.lang === "hi" ? 0.95 : 1.0;
    utter.pitch = 1.0;

    if (next.lang === "hi") {
      utter.lang = "hi-IN";
      // assign a SpeechSynthesisVoice or null
      utter.voice = hasHindiVoice ? voiceHi.current : voiceEn.current;
    } else {
      utter.lang = "en-US";
      utter.voice = voiceEn.current;
    }

    utter.onend = () => {
      isSpeaking.current = false;
      setTimeout(processSpeechQueue, 60);
    };
    utter.onerror = () => {
      isSpeaking.current = false;
      setTimeout(processSpeechQueue, 150);
    };

    // speak
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utter);
  };

  const enqueue = (text: string, lang: Lang, priority = false) => {
    if (!text) return;
    const now = Date.now();
    // avoid repeats & too frequent announcements
    if (text === lastSpokenText.current && now - lastAnnounceTime.current < 3000) return;
    if (!priority && now - lastAnnounceTime.current < 1200) return;

    lastAnnounceTime.current = now;
    if (priority) {
      window.speechSynthesis.cancel();
      speechQueue.current = [{ text, lang }];
      isSpeaking.current = false;
      setTimeout(processSpeechQueue, 40);
      return;
    }

    // keep queue short
    while (speechQueue.current.length > 6) speechQueue.current.shift();
    speechQueue.current.push({ text, lang });
    setTimeout(processSpeechQueue, 25);
  };

  const enqueueSpeakBoth = (en: string, hi?: string, priority = false) => {
    if (selectedLanguage === "en") enqueue(en, "en", priority);
    else if (selectedLanguage === "hi") enqueue(hi || en, "hi", priority);
    else {
      if (hi) enqueue(hi, "hi", priority);
      enqueue(en, "en", priority);
    }
  };

  // Replace earlier speakStatus with enqueue-based method (keeps original behaviour)
  const speakStatus = (msg: string) => {
    // speak short status message in selected language(s)
    const hi = {
      "Camera started. Position the book page in view.": "कैमरा चालू हुआ। पृष्ठ कैमरे में रखें।",
      "Stopped.": "रोक दिया गया।",
      "Full page detected. Starting to read.": "पूरा पृष्ठ मिला। पढ़ना शुरू कर रहा हूँ।",
      "No text detected. Move closer or improve lighting.": "कोई टेक्स्ट नहीं मिला। पास आएँ या रोशनी बढ़ाएँ।",
    } as Record<string, string>;

    const hiMsg = hi[msg] || ""; // if we have translation
    enqueueSpeakBoth(msg, hiMsg || undefined, true);
  };

  // Replace earlier speakFullText with chunked multi-language reader
  const speakFullText = (text: string) => {
    if (!text || text.length < 3) return;
    // if both, read Hindi translation first by attempting auto-translation for short lines (best-effort),
    // but to avoid changing logic we will speak the English text in Hindi voice only if selectedLanguage==='hi' or 'both' AND a Hindi voice is present.
    const chunks = chunkText(text, 180);
    // if user selected 'hi', attempt to read the original text with hi voice (may sound odd for English text),
    // better approach is to read English in English and only use Hindi voice for status / preset messages.
    // We'll follow this rule: content will be spoken in EN if selectedLanguage==='en', in HI if 'hi' (use same English text but hi voice) and both -> hi then en.
    const speakChunks = (lang: Lang, startIndex = 0) => {
      let i = startIndex;
      const speakNext = () => {
        if (i >= chunks.length) return;
        enqueue(chunks[i], lang);
        i++;
        // schedule next after previous finishes via queue processing
        setTimeout(() => speakNext(), 800); // small cadence; queue manager handles real timing
      };
      speakNext();
    };

    if (selectedLanguage === "en") {
      speakChunks("en", 0);
    } else if (selectedLanguage === "hi") {
      // try Hindi voice (reading English text) — better than silence; if you want real Hindi TTS via cloud, tell me.
      speakChunks("hi", 0);
    } else {
      // both: Hindi first, then English
      speakChunks("hi", 0);
      // delay en a bit so hindi starts first — queue system will handle order, but add small delay
      setTimeout(() => speakChunks("en", 0), 600);
    }
  };

  // ---- Tesseract OCR flow (unchanged logic, only replaced speakStatus/speakFullText calls) ----

  const startCamera = async () => {
    if (typeof navigator === "undefined" || !navigator.mediaDevices) {
      setStatus("❌ Camera not supported.");
      return;
    }

    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment",
          width: { ideal: 3840, max: 3840 },
          height: { ideal: 2160, max: 2160 },
        },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.play();
      }

      setStream(mediaStream);
      setIsCameraActive(true);
      setStatus("✅ Camera ready. Sweep slowly over the page.");
      enqueueSpeakBoth("Camera started. Position the book page in view.", "कैमरा चालू हुआ। पृष्ठ कैमरे में रखें।", true);
    } catch (err) {
      setStatus(`⚠ Error: ${(err as Error).message}`);
      enqueueSpeakBoth("Camera error", "कैमरा त्रुटि", true);
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
    setIsAutoScan(false);
    setIsBlindMode(false);
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    window.speechSynthesis.cancel();
    setStatus("🛑 Stopped.");
    enqueueSpeakBoth("Stopped.", "रोक दिया गया।", true);
  };

  const performOCR = useCallback(
    async (video: HTMLVideoElement, canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D) => {
      if (!video.videoWidth || !video.videoHeight || isProcessing) return;

      setIsProcessing(true);

      try {
        const processedImage = preprocessImage(ctx, canvas, video);

        const {
          data: { text, confidence: conf },
        } = await Tesseract.recognize(processedImage, "eng", tesseractConfig);

        const cleanText = text.trim().replace(/\s+/g, " ");

        if (cleanText && cleanText !== "No text detected") {
          if (cleanText !== previousTextRef.current && cleanText.length > previousTextRef.current.length * 0.8) {
            accumulatedTextRef.current += " " + cleanText;
            previousTextRef.current = cleanText;
            scanCountRef.current = 0;
          } else {
            scanCountRef.current++;
          }

          setExtractedText(accumulatedTextRef.current);
          setConfidence(conf);

          if (textAreaRef.current) {
            textAreaRef.current.focus();
          }

          if (scanCountRef.current >= 3 && conf > 75) {
            // Auto-read when stable and confident
            speakFullText(accumulatedTextRef.current);
            setStatus(`✅ Full page scanned! Reading... (Conf: ${Math.round(conf)}%)`);
            enqueueSpeakBoth("Full page detected. Starting to read.", "पूरा पृष्ठ मिला। पढ़ना शुरू कर रहा हूँ।", true);
            if (isBlindMode) setIsBlindMode(false);
          } else {
            setStatus(`🔍 Scanning... (${scanCountRef.current}/3 stable, Conf: ${Math.round(conf)}%)`);
            // Also short status in selected language
            speakStatus(`🔍 Scanning... (${scanCountRef.current}/3 stable, Conf: ${Math.round(conf)}%)`);
          }
        } else {
          setStatus("❌ No text. Adjust position.");
          enqueueSpeakBoth("No text detected. Move closer or improve lighting.", "कोई टेक्स्ट नहीं मिला। पास आएँ या रोशनी बढ़ाएँ।", true);
        }
      } catch (err) {
        setStatus(`❌ Error: ${(err as Error).message}`);
        enqueueSpeakBoth("Scanning failed.", "स्कैन विफल हुआ।", true);
      } finally {
        setIsProcessing(false);
      }
    },
    [isProcessing, isBlindMode, selectedLanguage] // keep same behaviour, added selectedLanguage dependency for speak calls
  );

  const captureAndRead = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    accumulatedTextRef.current = "";
    scanCountRef.current = 0;
    previousTextRef.current = "";
    performOCR(videoRef.current, canvas, ctx);
  };

  const toggleBlindMode = () => {
    setIsBlindMode(!isBlindMode);
    if (!isBlindMode && isCameraActive) {
      accumulatedTextRef.current = "";
      scanCountRef.current = 0;
      previousTextRef.current = "";
      let lastScan = 0;
      const scanLoop = (currentTime: number) => {
        if (currentTime - lastScan > 1000) {
          if (videoRef.current && canvasRef.current) {
            const canvas = canvasRef.current;
            const ctx = canvas.getContext("2d");
            if (ctx) {
              performOCR(videoRef.current!, canvas, ctx);
              lastScan = currentTime;
            }
          }
        }
        if (isBlindMode) {
          rafRef.current = requestAnimationFrame(scanLoop);
        }
      };
      rafRef.current = requestAnimationFrame(scanLoop);
      setStatus("👁 Blind Mode: Sweep over the page slowly. Auto-reads when full.");
      enqueueSpeakBoth("Blind mode on. Sweep the camera slowly over the entire page. It will read when complete.", "ब्लाइंड मोड चालू। धीरे-धीरे पूरे पृष्ठ पर कैमरा घुमाएँ। पूरा होने पर पढ़ेगा।", true);
    } else {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      setStatus("⏸ Blind Mode off.");
      enqueueSpeakBoth("Blind mode off.", "ब्लाइंड मोड बंद।", true);
    }
  };

  // video ready
  useEffect(() => {
    const video = videoRef.current;
    if (video) {
      const handleReady = () => {
        if (canvasRef.current) {
          canvasRef.current.width = video.videoWidth;
          canvasRef.current.height = video.videoHeight;
        }
      };
      video.addEventListener("loadedmetadata", handleReady);
      video.addEventListener("canplay", handleReady);
      return () => {
        video.removeEventListener("loadedmetadata", handleReady);
        video.removeEventListener("canplay", handleReady);
      };
    }
  }, []);

  // cleanup
  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (stream) stream.getTracks().forEach((track) => track.stop());
      window.speechSynthesis.cancel();
    };
  }, [stream]);

  // voices loaded fallback
  useEffect(() => {
    if ("speechSynthesis" in window && speechSynthesis.getVoices().length === 0) {
      speechSynthesis.onvoiceschanged = () => {
        loadVoices();
      };
    }
  }, []);

  // Test voices button handler
  const testVoices = () => {
    enqueueSpeakBoth("This is an English test.", "यह एक हिन्दी परीक्षण है।", true);
  };

  // on text-area blur => speak
  const handleTextBlur = () => {
    speakFullText(extractedText);
    enqueueSpeakBoth("Speaking selected text.", "चयनित पाठ बोल रहा हूँ।", true);
  };

  return (
    <main className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
      <div className="max-w-md w-full space-y-6">
        <h1 className="text-3xl font-bold text-center text-gray-800">📚 Blind-Friendly Page Scanner</h1>

        <p className="text-center text-gray-600">Auto-scans full pages as you sweep—no need to know size. Audio feedback guides you. Blur text to speak.</p>

        <div className="space-y-4">
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className="w-full max-w-md mx-auto rounded-lg shadow-md border-2 border-gray-300"
          />
          <canvas ref={canvasRef} className="hidden" />
        </div>

        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <button
            onClick={startCamera}
            disabled={isCameraActive}
            className="px-6 py-3 bg-green-600 text-white font-semibold rounded-lg shadow-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Start Camera
          </button>

          <button
            onClick={captureAndRead}
            disabled={!isCameraActive || isProcessing}
            className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex-1"
          >
            {isProcessing ? "Scanning..." : "Quick Scan"}
          </button>

          <button
            onClick={stopCamera}
            disabled={!isCameraActive}
            className="px-6 py-3 bg-red-600 text-white font-semibold rounded-lg shadow-md hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Stop All
          </button>
        </div>

        {isCameraActive && (
          <div className="space-y-2">
            <div className="flex justify-center">
              <button
                onClick={toggleBlindMode}
                disabled={isProcessing}
                className={`px-6 py-3 font-semibold rounded-lg shadow-md transition-colors ${
                  isBlindMode ? "bg-purple-600 text-white hover:bg-purple-700" : "bg-indigo-500 text-white hover:bg-indigo-600"
                }`}
              >
                {isBlindMode ? "Stop Sweep" : "Start Blind Sweep"}
              </button>
            </div>
            <p className="text-xs text-center text-gray-500">Sweep camera slowly over the whole page—it auto-detects & reads.</p>
          </div>
        )}

        <div className="flex items-center justify-center gap-3">
          <div className="bg-gray-900 px-3 py-2 rounded">
            <label className="mr-2 text-sm text-white">Language</label>
            <select value={selectedLanguage} onChange={(e) => setSelectedLanguage(e.target.value as any)} className="bg-black text-white px-2 rounded">
              <option value="en">English</option>
              <option value="hi">हिन्दी</option>
              <option value="both">Both</option>
            </select>
          </div>

          <button onClick={testVoices} className="bg-blue-600 px-3 py-2 rounded text-white">
            🔊 Test Voices
          </button>

          <div className="text-xs text-gray-700 ml-3">Hindi voice available: {hasHindiVoice ? "Yes" : "No"}</div>
        </div>

        <div className="text-center space-y-1">
          <p className="font-semibold text-gray-700">{status}</p>
          {confidence > 0 && <p className="text-xs text-blue-600">Conf: {Math.round(confidence)}%</p>}
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
          <h2 className="text-lg font-semibold mb-2 text-gray-800">Accumulated Text (Blur to Speak):</h2>
          <textarea
            ref={textAreaRef}
            value={extractedText}
            onBlur={handleTextBlur}
            readOnly
            className="w-full h-48 p-3 text-sm font-mono bg-gray-50 rounded resize-none border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Extracted text will appear here..."
          />
        </div>

        <div className="text-xs text-gray-500 text-center space-y-1">
          <p>👁 Blur text area to speak generated content. Blind Mode: Continuous scan + audio cues.</p>
          <p>💡 Enhanced accuracy: LSTM + median filter + 300 DPI resize. Deploy on Vercel for mobile.</p>
          <p>If Hindi still sounds English-accented: install a Hindi voice on your OS (Windows/macOS/Android) or tell me and I'll show a cloud-TTS option.</p>
        </div>
      </div>
    </main>
  );
}
