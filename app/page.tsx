"use client";

import { useEffect, useRef, useState } from "react";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";

interface DetectedObject {
  bbox: [number, number, number, number];
  class: string;
  score: number;
}

type Lang = "en" | "hi";

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [status, setStatus] = useState("Waiting for camera permission...");
  const [isStarted, setIsStarted] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState<"en" | "hi" | "both">("en");

  const voiceEn = useRef<SpeechSynthesisVoice | null>(null);
  const voiceHi = useRef<SpeechSynthesisVoice | null>(null);
  const hasHindiVoice = useRef(false);

  const DETECTION_INTERVAL_MS = 500;
  const CONFIDENCE_THRESHOLD = 0.45;

  const lastSpoken = useRef("");
  const isSpeaking = useRef(false);
  const speechQueue = useRef<{ text: string; lang: Lang }[]>([]);
  const lastAnnouncementTime = useRef(0);

  // === 1️⃣ Load Voices and Check Hindi Support ===
  useEffect(() => {
    const loadVoices = () => {
      const voices = window.speechSynthesis.getVoices();

      voiceEn.current =
        voices.find((v) => v.lang?.toLowerCase().startsWith("en") && v.name?.includes("Google")) ||
        voices.find((v) => v.lang?.toLowerCase().startsWith("en")) ||
        null;

      voiceHi.current =
        voices.find((v) => v.lang?.toLowerCase().startsWith("hi")) ||
        voices.find((v) => v.name?.toLowerCase().includes("hindi")) ||
        null;

      hasHindiVoice.current = !!voiceHi.current;

      console.log("🟢 English Voice:", voiceEn.current?.name, voiceEn.current?.lang);
      console.log("🟠 Hindi Voice:", voiceHi.current?.name, voiceHi.current?.lang);
      console.log("✅ Hindi supported:", hasHindiVoice.current);
    };

    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;
  }, []);

  // === 2️⃣ Camera Setup ===
  const setupCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setIsStarted(true);
        setStatus("Camera started. Detecting objects...");
        enqueueSpeakBoth("Camera activated. I will announce objects.", "कैमरा चालू हुआ। मैं वस्तुओं के बारे में बताऊँगा।");
      }
    } catch {
      setStatus("Camera permission denied or unavailable.");
    }
  };

  // === 3️⃣ Speech Engine (with fallback for Hindi) ===
  const processSpeechQueue = () => {
    if (isSpeaking.current) return;
    const next = speechQueue.current.shift();
    if (!next) return;
    isSpeaking.current = true;
    lastSpoken.current = next.text;

    const utter = new SpeechSynthesisUtterance(next.text);
    utter.rate = 1.0;
    utter.pitch = 1.0;

    if (next.lang === "hi") {
      utter.lang = "hi-IN";
      utter.voice = hasHindiVoice.current ? voiceHi.current : voiceEn.current;
    } else {
      utter.lang = "en-US";
      utter.voice = voiceEn.current;
    }

    utter.onend = () => {
      isSpeaking.current = false;
      setTimeout(processSpeechQueue, 80);
    };
    utter.onerror = () => {
      isSpeaking.current = false;
      setTimeout(processSpeechQueue, 150);
    };

    window.speechSynthesis.speak(utter);
  };

  const enqueue = (text: string, lang: Lang, priority = false) => {
    const now = Date.now();
    if (text === lastSpoken.current || now - lastAnnouncementTime.current < 1200) return;

    if (priority) {
      window.speechSynthesis.cancel();
      speechQueue.current = [{ text, lang }];
      isSpeaking.current = false;
      setTimeout(processSpeechQueue, 40);
    } else {
      speechQueue.current.push({ text, lang });
      setTimeout(processSpeechQueue, 40);
    }

    lastAnnouncementTime.current = now;
  };

  const enqueueSpeakBoth = (en: string, hi?: string, priority = false) => {
    if (selectedLanguage === "en") enqueue(en, "en", priority);
    else if (selectedLanguage === "hi") enqueue(hi || en, "hi", priority);
    else {
      enqueue(hi || en, "hi", priority);
      enqueue(en, "en", priority);
    }
  };

  // === 4️⃣ Load TensorFlow model ===
  useEffect(() => {
    (async () => {
      try {
        setStatus("Loading TensorFlow...");
        await tf.setBackend("webgl");
        await tf.ready();
        const detectionModel = await cocoSsd.load({ base: "mobilenet_v2" });
        setModel(detectionModel);
        setStatus("Model loaded. Tap 'Start Camera'.");
      } catch {
        setStatus("Model load failed.");
      }
    })();
  }, []);

  // === 5️⃣ Object Detection Loop ===
  useEffect(() => {
    if (!model || !isStarted) return;
    const detect = async () => {
      if (!videoRef.current) return;
      const preds = (await model.detect(videoRef.current)) as DetectedObject[];
      const best = preds.sort((a, b) => b.score - a.score)[0];
      if (best && best.score > CONFIDENCE_THRESHOLD) {
        const label = best.class;
        const hiLabel = translateToHindi(label);
        const enMsg = `Near you is a ${label}.`;
        const hiMsg = `पास में ${hiLabel} है।`;
        setStatus(selectedLanguage === "hi" ? hiMsg : enMsg);
        enqueueSpeakBoth(enMsg, hiMsg);
      } else {
        setStatus("Nothing detected");
      }
      requestAnimationFrame(detect);
    };
    detect();
  }, [model, isStarted, selectedLanguage]);

  // === 6️⃣ Translation helper ===
  const translateToHindi = (label: string) => {
    const map: Record<string, string> = {
      person: "व्यक्ति",
      bottle: "बोतल",
      cup: "कप",
      dog: "कुत्ता",
      cat: "बिल्ली",
      car: "गाड़ी",
      laptop: "लैपटॉप",
      phone: "फ़ोन",
    };
    return map[label] || label;
  };

  // === 7️⃣ UI ===
  return (
    <main className="flex flex-col items-center justify-center min-h-screen bg-black text-green-400 p-4">
      <h1 className="text-3xl font-bold mb-4">👁 Blind Vision Assistant (Auto Hindi Detection)</h1>

      <div className="flex gap-3 mb-4">
        <button onClick={setupCamera} className="bg-green-500 hover:bg-green-600 text-black px-4 py-2 rounded-xl font-semibold">
          🎥 Start Camera
        </button>

        <select
          value={selectedLanguage}
          onChange={(e) => setSelectedLanguage(e.target.value as "en" | "hi" | "both")}
          className="bg-gray-900 text-white px-3 py-2 rounded-xl"
        >
          <option value="en">English</option>
          <option value="hi">हिन्दी</option>
          <option value="both">Both</option>
        </select>
      </div>

      <video ref={videoRef} className="w-11/12 max-w-lg rounded-2xl border-2 border-green-400" autoPlay playsInline muted />
      <p className="mt-4 text-lg text-white text-center">{status}</p>

      <p className="mt-2 text-sm text-gray-400 text-center">
        🎤 Say “What do you see?” or “Stop speaking” — Auto-detects Hindi voice if available.
      </p>
    </main>
  );
}
