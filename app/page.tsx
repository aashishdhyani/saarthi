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
  const [isBlindMode, setIsBlindMode] = useState(false);
  const [confidence, setConfidence] = useState(0);
  const rafRef = useRef<number | null>(null);
  const previousTextRef = useRef('');
  const scanCountRef = useRef(0);
  const accumulatedTextRef = useRef('');

  type Lang = "en" | "hi";
  const [selectedLanguage, setSelectedLanguage] = useState<"en" | "hi" | "both">("en");
  const voiceEn = useRef<SpeechSynthesisVoice | null>(null);
  const voiceHi = useRef<SpeechSynthesisVoice | null>(null);
  const [hasHindiVoice, setHasHindiVoice] = useState(false);

  const speechQueue = useRef<{ text: string; lang: Lang }[]>([]);
  const isSpeaking = useRef(false);
  const lastAnnounceTime = useRef(0);
  const lastSpokenText = useRef("");

  const tesseractConfig = {
    logger: (m: any) => console.log(m),
    oem: 1 as const,
    psm: 6 as const,
    tessedit_char_blacklist: '|{}()[]' as const,
    tessedit_create_pdf: '0' as const,
  };

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

  // ---------- IMAGE PREPROCESSING ----------
  const calculateOtsuThreshold = (data: Uint8ClampedArray, width: number, height: number): number => {
    const histogram = new Array(256).fill(0);
    for (let i = 0; i < data.length; i += 4) histogram[data[i]]++;
    let total = width * height;
    let sum = 0;
    for (let i = 0; i < 256; i++) sum += i * histogram[i];
    let sumB = 0, wB = 0, wF = 0, max = 0, threshold = 0;
    for (let t = 0; t < 256; t++) {
      wB += histogram[t];
      if (wB === 0) continue;
      wF = total - wB;
      if (wF === 0) break;
      sumB += t * histogram[t];
      const mB = sumB / wB;
      const mF = (sum - sumB) / wF;
      const between = wB * wF * (mB - mF) * (mB - mF);
      if (between > max) { max = between; threshold = t; }
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

    let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
      const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
      data[i] = data[i + 1] = data[i + 2] = avg;
    }
    ctx.putImageData(imageData, 0, 0);

    // Median filter (3x3)
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

    imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    data = imageData.data;
    const threshold = calculateOtsuThreshold(data, canvas.width, canvas.height);
    for (let i = 0; i < data.length; i += 4) {
      const pixel = data[i];
      data[i] = data[i + 1] = data[i + 2] = pixel > threshold ? 255 : 0;
    }
    ctx.putImageData(imageData, 0, 0);

    ctx.filter = "contrast(1.2) brightness(1.1)";
    ctx.drawImage(canvas, 0, 0);
    ctx.filter = "none";

    ctx.strokeStyle = "white";
    ctx.lineWidth = 3;
    ctx.strokeRect(1, 1, canvas.width - 2, canvas.height - 2);
    return canvas.toDataURL("image/png");
  };

  // ---------- SPEECH SYSTEM ----------
  const loadVoices = () => {
    if (!("speechSynthesis" in window)) return;
    const voices = speechSynthesis.getVoices() || [];
    voiceEn.current =
      voices.find((v) => v.lang?.toLowerCase().startsWith("en") && v.name?.toLowerCase().includes("google")) ||
      voices.find((v) => v.lang?.toLowerCase().startsWith("en")) || null;
    voiceHi.current =
      voices.find((v) => v.lang?.toLowerCase().startsWith("hi")) ||
      voices.find((v) => v.name?.toLowerCase().includes("hindi")) || null;
    setHasHindiVoice(!!voiceHi.current);
  };

  useEffect(() => {
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
    utter.lang = next.lang === "hi" ? "hi-IN" : "en-US";
    utter.voice = next.lang === "hi" && hasHindiVoice ? voiceHi.current : voiceEn.current;

    utter.onend = () => {
      isSpeaking.current = false;
      setTimeout(processSpeechQueue, 40);
    };
    utter.onerror = () => {
      isSpeaking.current = false;
      setTimeout(processSpeechQueue, 100);
    };
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utter);
  };

  const enqueue = (text: string, lang: Lang, priority = false) => {
    if (!text) return;
    const now = Date.now();
    if (text === lastSpokenText.current && now - lastAnnounceTime.current < 3000) return;
    if (!priority && now - lastAnnounceTime.current < 1200) return;

    lastAnnounceTime.current = now;
    if (priority) {
      window.speechSynthesis.cancel();
      speechQueue.current = [{ text, lang }];
      isSpeaking.current = false;
      setTimeout(processSpeechQueue, 15);
      return;
    }

    while (speechQueue.current.length > 6) speechQueue.current.shift();
    speechQueue.current.push({ text, lang });
    setTimeout(processSpeechQueue, 15);
  };

  const enqueueSpeakBoth = (en: string, hi?: string, priority = false) => {
    if (selectedLanguage === "en") enqueue(en, "en", priority);
    else if (selectedLanguage === "hi") enqueue(hi || en, "hi", priority);
    else {
      if (hi) enqueue(hi, "hi", priority);
      setTimeout(() => enqueue(en, "en", priority), 400); // 400 ms delay
    }
  };

  const speakFullText = (text: string) => {
    if (!text || text.length < 3) return;
    const chunks = chunkText(text, 180);
    const speakChunks = (lang: Lang, i = 0) => {
      if (i >= chunks.length) return;
      enqueue(chunks[i], lang);
      setTimeout(() => speakChunks(lang, i + 1), 800);
    };
    if (selectedLanguage === "en") speakChunks("en");
    else if (selectedLanguage === "hi") speakChunks("hi");
    else {
      speakChunks("hi");
      setTimeout(() => speakChunks("en"), 400);
    }
  };

  // ---------- CAMERA + OCR ----------
  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 3840 }, height: { ideal: 2160 } },
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
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    setStream(null);
    setIsCameraActive(false);
    setIsBlindMode(false);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    window.speechSynthesis.cancel();
    setStatus("🛑 Stopped.");
    enqueueSpeakBoth("Stopped.", "रोक दिया गया।", true);
  };

  const performOCR = useCallback(async (video: HTMLVideoElement, canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D) => {
    if (!video.videoWidth || !video.videoHeight || isProcessing) return;
    setIsProcessing(true);
    try {
      const processed = preprocessImage(ctx, canvas, video);
      const { data: { text, confidence: conf } } = await Tesseract.recognize(processed, "eng", tesseractConfig);
      const cleanText = text.trim().replace(/\s+/g, " ");
      if (cleanText && cleanText !== "No text detected") {
        if (cleanText !== previousTextRef.current && cleanText.length > previousTextRef.current.length * 0.8) {
          accumulatedTextRef.current += " " + cleanText;
          previousTextRef.current = cleanText;
          scanCountRef.current = 0;
        } else scanCountRef.current++;
        setExtractedText(accumulatedTextRef.current);
        setConfidence(conf);
        if (scanCountRef.current >= 3 && conf > 75) {
          speakFullText(accumulatedTextRef.current);
          setStatus(`✅ Full page scanned! Reading... (${Math.round(conf)}%)`);
          enqueueSpeakBoth("Full page detected. Starting to read.", "पूरा पृष्ठ मिला। पढ़ना शुरू कर रहा हूँ।", true);
        } else {
          setStatus(`🔍 Scanning... (${scanCountRef.current}/3 stable, ${Math.round(conf)}%)`);
          // short status audio optionally
          // enqueueSpeakBoth(`Scanning ${scanCountRef.current} of 3`, `स्कैन ${scanCountRef.current} में से 3`, false);
        }
      } else {
        setStatus("❌ No text. Adjust position.");
        enqueueSpeakBoth("No text detected. Move closer or improve lighting.", "कोई टेक्स्ट नहीं मिला। पास आएँ या रोशनी बढ़ाएँ।", true);
      }
    } catch (e) {
      setStatus(`❌ Error: ${(e as Error).message}`);
      enqueueSpeakBoth("Scanning failed.", "स्कैन विफल हुआ।", true);
    } finally {
      setIsProcessing(false);
    }
  }, [isProcessing, selectedLanguage]);

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
      const loop = (t: number) => {
        if (t - lastScan > 1000 && videoRef.current && canvasRef.current) {
          const ctx = canvasRef.current.getContext("2d");
          if (ctx) performOCR(videoRef.current, canvasRef.current, ctx);
          lastScan = t;
        }
        if (isBlindMode) rafRef.current = requestAnimationFrame(loop);
      };
      rafRef.current = requestAnimationFrame(loop);
      enqueueSpeakBoth("Blind mode on. Sweep slowly; reading when full.", "ब्लाइंड मोड चालू। धीरे-धीरे स्कैन करें, पूरा होने पर पढ़ेगा।", true);
      setStatus("👁 Blind Mode: Sweep over the page slowly. Auto-reads when full.");
    } else {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      enqueueSpeakBoth("Blind mode off.", "ब्लाइंड मोड बंद।", true);
      setStatus("⏸ Blind Mode off.");
    }
  };

  const handleTextBlur = () => {
    speakFullText(extractedText);
    enqueueSpeakBoth("Speaking selected text.", "चयनित पाठ बोल रहा हूँ।", true);
  };

  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (stream) stream.getTracks().forEach((t) => t.stop());
      window.speechSynthesis.cancel();
    };
  }, [stream]);

  // ensure voices load
  useEffect(() => {
    if ("speechSynthesis" in window && speechSynthesis.getVoices().length === 0) {
      speechSynthesis.onvoiceschanged = () => loadVoices();
    }
  }, []);

  // Test voices
  const testVoices = () => {
    enqueueSpeakBoth("This is an English test.", "यह एक हिन्दी परीक्षण है।", true);
  };

  return (
    <main className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
      <div className="max-w-md w-full space-y-6">
        <h1 className="text-3xl font-bold text-center text-gray-800">📚 Blind-Friendly Page Scanner</h1>
        <p className="text-center text-gray-600">Auto-scans pages. Hindi + English voice with faster speech sync (400 ms).</p>

        <video ref={videoRef} autoPlay muted playsInline className="w-full max-w-md mx-auto rounded-lg shadow-md border-2 border-gray-300" />
        <canvas ref={canvasRef} className="hidden" />

        <div className="flex flex-wrap gap-3 justify-center">
          <button onClick={startCamera} disabled={isCameraActive} className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50">
            Start Camera
          </button>
          <button onClick={captureAndRead} disabled={!isCameraActive || isProcessing} className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50">
            {isProcessing ? "Scanning..." : "Quick Scan"}
          </button>
          <button onClick={stopCamera} disabled={!isCameraActive} className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50">
            Stop All
          </button>
          <button onClick={toggleBlindMode} disabled={!isCameraActive || isProcessing} className="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50">
            {isBlindMode ? "Stop Sweep" : "Start Blind Sweep"}
          </button>
        </div>

        <div className="flex items-center justify-center gap-3">
          <div className="bg-gray-900 px-3 py-2 rounded">
            <label className="mr-2 text-sm text-white">Language</label>
            <select value={selectedLanguage} onChange={(e) => setSelectedLanguage(e.target.value as any)} className="bg-black text-white px-2 rounded">
              <option value="en">English</option>
              <option value="hi">हिन्दी</option>
              <option value="both">Both</option>
            </select>
          </div>

          <button onClick={testVoices} className="bg-blue-600 px-3 py-2 rounded text-white">🔊 Test Voices</button>

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
