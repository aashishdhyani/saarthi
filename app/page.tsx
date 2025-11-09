// app/page.tsx
"use client";

import { useEffect, useRef, useState } from "react";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";

interface DetectedObject {
  bbox: [number, number, number, number]; // [x, y, w, h]
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

  // voice state
  const voiceEn = useRef<SpeechSynthesisVoice | null>(null);
  const voiceHi = useRef<SpeechSynthesisVoice | null>(null);
  const [hasHindiVoice, setHasHindiVoice] = useState(false);

  // speech queue
  const speechQueue = useRef<{ text: string; lang: Lang }[]>([]);
  const isSpeaking = useRef(false);
  const lastAnnounceTime = useRef(0);
  const lastAnnouncedText = useRef("");

  // detection tuning
  const DETECTION_INTERVAL_MS = 600; // tune to reduce chatter
  const CONFIDENCE_THRESHOLD = 0.45;
  const STABILITY_WINDOW = 6; // how many recent labels to consider for stability
  const MIN_ANNOUNCE_GAP = 1800; // ms between announcements

  // simple distance estimation params
  const focalPx = 700; // kept simple; seefor calibration if you want better accuracy
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

  // detection history for stability + averaging distance
  const recentLabels = useRef<string[]>([]);
  const recentDistances = useRef<number[]>([]);

  // ====== Load voices and detect Hindi availability ======
  useEffect(() => {
    const load = () => {
      const voices = window.speechSynthesis.getVoices() || [];

      // prefer Google voices where possible, else any en / hi
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

    load();
    window.speechSynthesis.onvoiceschanged = load;
  }, []);

  // ====== Speech queue manager ======
  const processSpeechQueue = () => {
    if (isSpeaking.current) return;
    const next = speechQueue.current.shift();
    if (!next) return;
    isSpeaking.current = true;
    lastAnnouncedText.current = next.text;

    const utt = new SpeechSynthesisUtterance(next.text);
    utt.rate = 1.0;
    utt.pitch = 1.0;

    if (next.lang === "hi") {
      utt.lang = "hi-IN";
      utt.voice = hasHindiVoice ? voiceHi.current : voiceEn.current;
    } else {
      utt.lang = "en-US";
      utt.voice = voiceEn.current;
    }

    utt.onend = () => {
      isSpeaking.current = false;
      setTimeout(processSpeechQueue, 60);
    };
    utt.onerror = () => {
      isSpeaking.current = false;
      setTimeout(processSpeechQueue, 200);
    };
    window.speechSynthesis.speak(utt);
  };

  const enqueue = (text: string, lang: Lang, priority = false) => {
    if (!text) return;
    const now = Date.now();
    // prevent repeats and too-frequent speech
    if (text === lastAnnouncedText.current && now - lastAnnounceTime.current < 3000) return;
    if (!priority && now - lastAnnounceTime.current < MIN_ANNOUNCE_GAP) return;

    lastAnnounceTime.current = now;
    if (priority) {
      window.speechSynthesis.cancel();
      speechQueue.current = [{ text, lang }];
      isSpeaking.current = false;
      setTimeout(processSpeechQueue, 40);
      return;
    }

    // keep queue short
    while (speechQueue.current.length > 4) speechQueue.current.shift();
    speechQueue.current.push({ text, lang });
    setTimeout(processSpeechQueue, 20);
  };

  const enqueueSpeakBoth = (en: string, hi?: string, priority = false) => {
    if (selectedLanguage === "en") enqueue(en, "en", priority);
    else if (selectedLanguage === "hi") enqueue(hi || en, "hi", priority);
    else {
      // both: speak Hindi then English
      if (hi) enqueue(hi, "hi", priority);
      enqueue(en, "en", priority);
    }
  };

  // Test voices quick helper
  const testVoices = () => {
    enqueue("This is an English test.", "en", true);
    enqueue("यह एक हिन्दी परीक्षण है।", "hi", true);
  };

  // ====== Camera setup ======
  const setupCamera = async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } }, audio: false });
      if (videoRef.current) {
        videoRef.current.srcObject = s;
        await videoRef.current.play();
        setIsStarted(true);
        setStatus("Camera started. Detecting...");
        enqueueSpeakBoth("Camera activated. I will announce objects and distances.", "कैमरा चालू हुआ। मैं वस्तुओं और उनकी दूरी बताऊँगा।");
      }
    } catch (e) {
      console.error("cam error", e);
      setStatus("Camera permission denied or unavailable.");
    }
  };

  // ====== Load model ======
  useEffect(() => {
    (async () => {
      try {
        setStatus("Initializing TensorFlow...");
        await tf.setBackend("webgl");
        await tf.ready();
        setStatus("Loading COCO-SSD model...");
        const m = await cocoSsd.load({ base: "mobilenet_v2" });
        setModel(m);
        setStatus("Model loaded — press Start Camera.");
      } catch (e) {
        console.error(e);
        setStatus("Failed to load model.");
      }
    })();
  }, []);

  // ====== Distance estimate (simple) ======
  const estimateDistanceMeters = (label: string, bboxHeightPx: number) => {
    if (!bboxHeightPx || bboxHeightPx <= 0) return null;
    const realHeight = avgHeights[label] || 0.4;
    return (realHeight * focalPx) / bboxHeightPx;
  };

  // ====== Detection polling with stability/averaging ======
  useEffect(() => {
    if (!model || !isStarted) return;
    let stopped = false;

    const poll = async () => {
      if (stopped) return;
      if (!videoRef.current) {
        setTimeout(poll, DETECTION_INTERVAL_MS);
        return;
      }

      try {
        const preds = (await model.detect(videoRef.current)) as DetectedObject[];
        // choose top prediction above threshold
        const top = preds.sort((a, b) => b.score - a.score).find((p) => p.score > CONFIDENCE_THRESHOLD);
        if (top) {
          const label = top.class;
          const pxH = top.bbox[3];
          const dist = estimateDistanceMeters(label, pxH) ?? 0;

          // push to history
          recentLabels.current.push(label);
          if (recentLabels.current.length > STABILITY_WINDOW) recentLabels.current.shift();

          recentDistances.current.push(dist);
          if (recentDistances.current.length > STABILITY_WINDOW) recentDistances.current.shift();

          // determine stable label: majority in window
          const counts: Record<string, number> = {};
          for (const l of recentLabels.current) counts[l] = (counts[l] || 0) + 1;
          const majority = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
          const stableLabel = majority && majority[1] >= Math.ceil(STABILITY_WINDOW / 2) ? majority[0] : null;

          if (stableLabel) {
            // average distance for recent distances (exclude zeros)
            const valid = recentDistances.current.filter((d) => d && d > 0);
            const avgDist = valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : null;

            if (avgDist) {
              const dRounded = Math.round(avgDist * 10) / 10; // 0.1 m precision
              const enMsg = `A ${stableLabel} is about ${dRounded} meters away.`;
              const hiLabel = translateToHindi(stableLabel);
              const hiMsg = `पास में ${hiLabel} लगभग ${dRounded} मीटर दूर है।`;

              setStatus(selectedLanguage === "hi" ? hiMsg : enMsg);
              enqueueSpeakBoth(enMsg, hiMsg);
            } else {
              const enMsg = `Near you is a ${stableLabel}.`;
              const hiMsg = `पास में ${translateToHindi(stableLabel)} है।`;
              setStatus(selectedLanguage === "hi" ? hiMsg : enMsg);
              enqueueSpeakBoth(enMsg, hiMsg);
            }
          }
        } else {
          // no detection
          recentLabels.current = [];
          recentDistances.current = [];
          setStatus("Nothing detected");
        }
      } catch (err) {
        console.error("detect err", err);
      } finally {
        setTimeout(poll, DETECTION_INTERVAL_MS);
      }
    };

    poll();
    return () => {
      stopped = true;
    };
  }, [model, isStarted, selectedLanguage]);

  // translate helper
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
    <main className="flex flex-col items-center justify-start min-h-screen bg-black text-green-400 p-4 gap-4">
      <h1 className="text-2xl font-bold mt-2">👁 Blind Vision Assistant — Improved Speech & Distance</h1>

      <div className="flex gap-3 items-center">
        <button onClick={setupCamera} className="bg-green-500 px-4 py-2 rounded">
          🎥 Start Camera
        </button>

        <div className="bg-gray-900 px-3 py-2 rounded">
          <label className="mr-2 text-sm">Language</label>
          <select value={selectedLanguage} onChange={(e) => setSelectedLanguage(e.target.value as any)} className="bg-black text-white px-2 rounded">
            <option value="en">English</option>
            <option value="hi">हिन्दी</option>
            <option value="both">Both</option>
          </select>
        </div>

        <button onClick={testVoices} className="bg-blue-600 px-3 py-2 rounded text-white">
          🔊 Test Voices
        </button>

        <div className="text-xs text-gray-300 ml-3">Hindi voice available: {hasHindiVoice ? "Yes" : "No"}</div>
      </div>

      <div className="w-full max-w-xl">
        <video ref={videoRef} className="w-full rounded-xl border-2 border-green-400" autoPlay playsInline muted />
      </div>

      <div className="w-full max-w-xl bg-gray-900 p-3 rounded">
        <p className="text-white">Status: {status}</p>
        <p className="text-xs text-gray-400 mt-2">Notes: detection interval {DETECTION_INTERVAL_MS}ms — stable window {STABILITY_WINDOW} frames. Increase interval to reduce announcements.</p>
      </div>

      <p className="text-xs text-gray-400">If Hindi still sounds English-accented: install a Hindi voice on your OS (Windows / macOS / Android) or tell me and I’ll show a cloud TTS option.</p>
    </main>
  );
}
