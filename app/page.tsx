// app/page.tsx
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

  const lastSpoken = useRef("");
  const isSpeaking = useRef(false);
  const speechQueue = useRef<{ text: string; lang: Lang }[]>([]);
  const lastAnnouncementTime = useRef(0);

  const CONFIDENCE_THRESHOLD = 0.45;

  // load voices
  useEffect(() => {
    const loadVoices = () => {
      const voices = window.speechSynthesis.getVoices();
      voiceEn.current =
        voices.find((v) => v.lang?.toLowerCase().startsWith("en") && v.name?.toLowerCase().includes("google")) ||
        voices.find((v) => v.lang?.toLowerCase().startsWith("en")) ||
        null;

      voiceHi.current =
        voices.find((v) => v.lang?.toLowerCase().startsWith("hi")) ||
        voices.find((v) => v.name?.toLowerCase().includes("hindi")) ||
        null;

      hasHindiVoice.current = !!voiceHi.current;
      console.log("English voice:", voiceEn.current?.name);
      console.log("Hindi voice:", voiceHi.current?.name);
      console.log("Hindi supported:", hasHindiVoice.current);
    };

    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;
  }, []);

  // speech queue processor
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
      setTimeout(() => processSpeechQueue(), 60);
    };
    utter.onerror = () => {
      isSpeaking.current = false;
      setTimeout(() => processSpeechQueue(), 150);
    };

    window.speechSynthesis.speak(utter);
  };

  const enqueue = (text: string, lang: Lang, priority = false) => {
    if (!text) return;
    const now = Date.now();
    if (text === lastSpoken.current || now - lastAnnouncementTime.current < 1200) return;
    lastAnnouncementTime.current = now;

    if (priority) {
      window.speechSynthesis.cancel();
      speechQueue.current = [{ text, lang }];
      isSpeaking.current = false;
      setTimeout(processSpeechQueue, 40);
      return;
    }
    speechQueue.current.push({ text, lang });
    setTimeout(processSpeechQueue, 40);
  };

  const enqueueSpeakBoth = (en: string, hi?: string, priority = false) => {
    if (selectedLanguage === "en") enqueue(en, "en", priority);
    else if (selectedLanguage === "hi") enqueue(hi || en, "hi", priority);
    else {
      if (hi) enqueue(hi, "hi", priority);
      enqueue(en, "en", priority);
    }
  };

  // setup camera
  const setupCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setIsStarted(true);
        setStatus("Camera started. Detecting objects...");
        enqueueSpeakBoth("Camera activated. I will announce objects and distances.", "कैमरा चालू हुआ। मैं वस्तुओं और दूरी के बारे में बताऊँगा।");
      }
    } catch (err) {
      console.error("Camera error:", err);
      setStatus("Camera permission denied or unavailable.");
    }
  };

  // load TensorFlow model
  useEffect(() => {
    (async () => {
      try {
        setStatus("Initializing TensorFlow...");
        await tf.setBackend("webgl");
        await tf.ready();
        setStatus("Loading model...");
        const m = await cocoSsd.load({ base: "mobilenet_v2" });
        setModel(m);
        setStatus("Model loaded — press Start Camera.");
      } catch (err) {
        console.error(err);
        setStatus("Model failed to load.");
      }
    })();
  }, []);

  // estimate distance from bounding box height
  const estimateDistance = (label: string, height: number) => {
    if (height <= 0) return null;
    const avgHeights: Record<string, number> = {
      person: 1.7,
      bottle: 0.25,
      chair: 0.9,
      dog: 0.5,
      cat: 0.3,
      car: 1.5,
      laptop: 0.02,
      cup: 0.1,
    };
    const realHeight = avgHeights[label] || 0.4;
    const focalPx = 700; // estimated default focal length
    const distance = (realHeight * focalPx) / height;
    return distance;
  };

  // detection loop
  useEffect(() => {
    if (!model || !isStarted) return;
    let active = true;

    const run = async () => {
      if (!active) return;
      if (!videoRef.current) {
        requestAnimationFrame(run);
        return;
      }

      try {
        const preds = (await model.detect(videoRef.current)) as DetectedObject[];
        if (preds.length > 0) {
          const best = preds.sort((a, b) => b.score - a.score)[0];
          if (best.score > CONFIDENCE_THRESHOLD) {
            const label = best.class;
            const bboxHeight = best.bbox[3];
            const dist = estimateDistance(label, bboxHeight);
            const distRounded = dist ? Math.round(dist * 10) / 10 : null;

            const enMsg = distRounded
              ? `A ${label} is about ${distRounded} meters away.`
              : `Near you is a ${label}.`;
            const hiMsg = distRounded
              ? `पास में ${translateToHindi(label)} लगभग ${distRounded} मीटर दूर है।`
              : `पास में ${translateToHindi(label)} है।`;

            setStatus(selectedLanguage === "hi" ? hiMsg : enMsg);
            enqueueSpeakBoth(enMsg, hiMsg);
          }
        } else {
          setStatus("Nothing detected");
        }
      } catch (e) {
        console.error("Detection error:", e);
      }

      requestAnimationFrame(run);
    };

    run();
    return () => {
      active = false;
    };
  }, [model, isStarted, selectedLanguage]);

  const translateToHindi = (label: string) => {
    const map: Record<string, string> = {
      person: "व्यक्ति",
      bottle: "बोतल",
      chair: "कुर्सी",
      cup: "कप",
      dog: "कुत्ता",
      cat: "बिल्ली",
      car: "गाड़ी",
      laptop: "लैपटॉप",
      phone: "फ़ोन",
    };
    return map[label] || label;
  };

  return (
    <main className="flex flex-col items-center justify-center min-h-screen bg-black text-green-400 p-4">
      <h1 className="text-2xl font-bold mb-4">👁 Blind Vision Assistant (Distance + Hindi Voice)</h1>

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
        🎤 Detects objects, speaks in Hindi/English, and estimates distance (approx).
      </p>
    </main>
  );
}
