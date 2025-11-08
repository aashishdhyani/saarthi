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

type Lang = "en" | "hi" | "other";

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const detectionIntervalRef = useRef<number | null>(null);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [status, setStatus] = useState("Waiting for camera permission...");
  const [isStarted, setIsStarted] = useState(false);

  // language selection state
  const [selectedLanguage, setSelectedLanguage] = useState<"en" | "hi" | "both">("en");

  // voices found
  const voiceEn = useRef<SpeechSynthesisVoice | null>(null);
  const voiceHi = useRef<SpeechSynthesisVoice | null>(null);

  // Controls you can tune
  const DETECTION_INTERVAL_MS = 500; // how often to run detection (ms)
  const MIN_ANNOUNCEMENT_INTERVAL = 1200; // min time between announcements (ms)
  const STABILITY_REQUIRED_COUNT = 4; // number of identical labels required to be "stable"
  const CONFIDENCE_THRESHOLD = 0.45; // detection score threshold

  // internals
  const detectingRef = useRef(false);
  const recentLabels = useRef<string[]>([]);
  const lastAnnouncementTime = useRef(0);
  const lastSpoken = useRef("");
  // speechQueue items: { text, lang }
  const speechQueue = useRef<{ text: string; lang: Lang }[]>([]);
  const isSpeaking = useRef(false);

  // Basic Hindi translations for common COCO classes (extend as needed)
  const hiLabelMap: Record<string, string> = {
    person: "व्यक्ति",
    bottle: "बोतल",
    chair: "कुर्सी",
    couch: "सोफा",
    cup: "कप",
    dog: "कुत्ता",
    cat: "बिल्ली",
    book: "किताब",
    cell_phone: "मोबाइल",
    laptop: "लैपटॉप",
    backpack: "बैग",
    handbag: "हैंडबैग",
    umbrella: "छाता",
    table: "मेज़",
    car: "गाड़ी",
    bicycle: "साइकिल",
    motorcycle: "मोटरसाइकिल",
    tv: "टीवी",
    remote: "रिमोट",
    keyboard: "कीबोर्ड",
    mouse: "माउस",
    // add more as you encounter them
  };

  const translateLabelToHindi = (label: string) => {
    // model sometimes returns spaces; convert typical '_' to ' '
    const key = label.replace(/\s+/g, "_");
    return hiLabelMap[key] || label;
  };

  const setupCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setIsStarted(true);
        setStatus("Camera started. Detecting objects...");
        enqueueSpeakBoth("Camera activated. I will announce objects and people I detect.", "कैमरा चालू हुआ। मैं वस्तु और लोगों के बारे में बताऊँगा।");
      }
    } catch (err) {
      console.error("Camera error:", err);
      setStatus("Camera permission denied or unavailable.");
    }
  };

  // load voices (english & hindi if available)
  useEffect(() => {
    const loadVoice = () => {
      const voices = window.speechSynthesis.getVoices();
      if (!voices || voices.length === 0) return;
      voiceEn.current =
        voices.find((v) => v.lang?.startsWith("en") && v.name?.toLowerCase().includes("google")) ||
        voices.find((v) => v.lang?.startsWith("en")) ||
        voices[0] ||
        null;
      voiceHi.current =
        voices.find((v) => v.lang?.startsWith("hi")) ||
        voices.find((v) => v.lang?.startsWith("hi-IN")) ||
        voices.find((v) => v.name?.toLowerCase().includes("hindi")) ||
        null;

      // Debug prints optional
      console.log("Selected English voice:", voiceEn.current?.name, voiceEn.current?.lang);
      console.log("Selected Hindi voice:", voiceHi.current?.name, voiceHi.current?.lang);
    };
    loadVoice();
    window.speechSynthesis.onvoiceschanged = loadVoice;
  }, []);

  // speech queue manager ---------------------------------------------------
  const processSpeechQueue = () => {
    if (isSpeaking.current) return;
    const next = speechQueue.current.shift();
    if (!next) return;
    isSpeaking.current = true;
    lastSpoken.current = next.text;

    const utter = new SpeechSynthesisUtterance(next.text);
    utter.rate = 1.0;
    utter.pitch = 1.0;

    // assign voice based on language tag for this utterance
    if (next.lang === "hi" && voiceHi.current) utter.voice = voiceHi.current;
    else if (next.lang === "en" && voiceEn.current) utter.voice = voiceEn.current;
    // else allow browser to pick default voice

    utter.onend = () => {
      isSpeaking.current = false;
      setTimeout(() => processSpeechQueue(), 80);
    };
    utter.onerror = (e) => {
      console.error("Speech error:", e);
      isSpeaking.current = false;
      setTimeout(() => processSpeechQueue(), 150);
    };
    window.speechSynthesis.speak(utter);
  };

  // enqueue helpers
  const enqueue = (text: string, lang: Lang, priority = false) => {
    if (!text) return;
    // avoid immediate repeat
    if (text === lastSpoken.current) return;

    const now = Date.now();
    if (!priority && now - lastAnnouncementTime.current < MIN_ANNOUNCEMENT_INTERVAL) {
      return;
    }

    if (priority) {
      // clear and speak immediately (used for voice commands)
      window.speechSynthesis.cancel();
      speechQueue.current = [{ text, lang }];
      lastAnnouncementTime.current = now;
      isSpeaking.current = false;
      setTimeout(processSpeechQueue, 40);
      return;
    }

    speechQueue.current.push({ text, lang });
    lastAnnouncementTime.current = now;
    setTimeout(processSpeechQueue, 20);
  };

  // enqueue based on selectedLanguage: en | hi | both
  const enqueueSpeakBoth = (enText: string, hiText?: string, priority = false) => {
    const hi = hiText || ""; // hiText may be provided
    if (selectedLanguage === "en") {
      enqueue(enText, "en", priority);
    } else if (selectedLanguage === "hi") {
      enqueue(hi || enText, "hi", priority);
    } else {
      // both: speak Hindi first (if available), then English
      if (hi) enqueue(hi, "hi", priority);
      enqueue(enText, "en", priority);
    }
  };

  // load model
  useEffect(() => {
    (async () => {
      try {
        setStatus("Initializing TensorFlow...");
        await tf.setBackend("webgl");
        await tf.ready();
        setStatus("Loading COCO-SSD model...");
        const detectionModel = await cocoSsd.load({ base: "mobilenet_v2" });
        setModel(detectionModel);
        setStatus("Model loaded. Tap 'Start Camera'.");
      } catch (err) {
        console.error("Model load error:", err);
        setStatus("Failed to load model.");
      }
    })();
  }, []);

  // detection loop (interval-based, stable announcements) -------------------
  useEffect(() => {
    if (!model || !isStarted) return;

    // clear previous
    if (detectionIntervalRef.current) {
      window.clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }

    const runDetection = async () => {
      if (!videoRef.current || detectingRef.current || videoRef.current.readyState < 2) return;
      detectingRef.current = true;
      try {
        const preds = (await model.detect(videoRef.current)) as DetectedObject[];
        let label = "nothing";
        if (preds.length > 0) {
          const best = preds.sort((a, b) => b.score - a.score).find((p) => p.score > CONFIDENCE_THRESHOLD);
          if (best) label = best.class;
        }

        recentLabels.current.push(label);
        if (recentLabels.current.length > STABILITY_REQUIRED_COUNT * 2) recentLabels.current.shift();

        const tail = recentLabels.current.slice(-STABILITY_REQUIRED_COUNT);
        const stable = tail.every((v) => v === tail[0]) ? tail[0] : "nothing";

        if (stable !== "nothing") {
          // prepare both-language messages
          const enMsg = stable === "person" ? "A person is near you." : `Near you is a ${stable}.`;
          const hiLabel = translateLabelToHindi(stable);
          const hiMsg = stable === "person" ? "एक व्यक्ति आपके पास है।" : `पास में ${hiLabel} है।`;

          setStatus(selectedLanguage === "hi" ? hiMsg : enMsg);
          enqueueSpeakBoth(enMsg, hiMsg, false);
        } else {
          setStatus("Nothing detected");
        }
      } catch (e) {
        console.error("Detection error:", e);
      } finally {
        detectingRef.current = false;
      }
    };

    runDetection();
    detectionIntervalRef.current = window.setInterval(runDetection, DETECTION_INTERVAL_MS);

    return () => {
      if (detectionIntervalRef.current) {
        window.clearInterval(detectionIntervalRef.current);
        detectionIntervalRef.current = null;
      }
    };
  }, [model, isStarted, selectedLanguage]);

  // voice command listener (speech recognition)
  useEffect(() => {
    if (!("webkitSpeechRecognition" in window || "SpeechRecognition" in window)) {
      console.warn("Speech recognition not supported.");
      return;
    }
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = "en-US";

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      const transcript = event.results[event.results.length - 1][0].transcript
        .trim()
        .toLowerCase();
      console.log("Voice command:", transcript);

      if (transcript.includes("what do you see")) {
        // priority read current status in chosen language(s)
        // If both, speak Hindi then English
        const lastStatus = status || "I don't see anything right now.";
        // Prepare Hindi equivalent quick mapping for common phrases:
        // For simplicity, if status contains "A person" or "Near you is a", we try to map; otherwise fallback
        if (selectedLanguage === "en") {
          enqueue(lastStatus, "en", true);
        } else if (selectedLanguage === "hi") {
          // attempt a simple mapping for the two standard messages
          if (lastStatus.includes("person")) enqueue("एक व्यक्ति आपके पास है।", "hi", true);
          else if (lastStatus.includes("Near you is a")) {
            const label = lastStatus.replace("Near you is a ", "").replace(".", "").trim();
            enqueue(`पास में ${translateLabelToHindi(label)} है।`, "hi", true);
          } else {
            enqueue("मैं अभी कुछ नहीं देख रहा हूँ।", "hi", true);
          }
        } else {
          // both
          if (lastStatus.includes("person")) {
            enqueue("एक व्यक्ति आपके पास है।", "hi", true);
            enqueue("A person is near you.", "en", true);
          } else if (lastStatus.includes("Near you is a")) {
            const label = lastStatus.replace("Near you is a ", "").replace(".", "").trim();
            enqueue(`पास में ${translateLabelToHindi(label)} है।`, "hi", true);
            enqueue(lastStatus, "en", true);
          } else {
            enqueue("मैं अभी कुछ नहीं देख रहा हूँ।", "hi", true);
            enqueue("I don't see anything right now.", "en", true);
          }
        }
      } else if (transcript.includes("stop speaking")) {
        window.speechSynthesis.cancel();
        speechQueue.current = [];
        isSpeaking.current = false;
        lastSpoken.current = "";
      } else if (transcript.includes("start camera")) {
        setupCamera();
      } else if (transcript.includes("language english") || transcript.includes("set language english")) {
        setSelectedLanguage("en");
        enqueueSpeakBoth("Language set to English.", "भाषा अंग्रेज़ी select की गई।");
      } else if (transcript.includes("language hindi") || transcript.includes("set language hindi")) {
        setSelectedLanguage("hi");
        enqueueSpeakBoth("Language set to Hindi.", "भाषा हिन्दी चुनी गई।");
      } else if (transcript.includes("language both") || transcript.includes("set language both")) {
        setSelectedLanguage("both");
        enqueueSpeakBoth("Language set to both Hindi and English.", "भाषा दोनों (हिन्दी और अंग्रेज़ी) चुनी गई।");
      }
    };

    recognition.onerror = (err: any) => console.error("Speech recognition error:", err);
    recognition.onend = () => recognition.start();
    recognition.start();

    return () => {
      try {
        recognition.stop();
      } catch {}
    };
  }, [status, selectedLanguage]);

  // UI helpers
  const changeLanguage = (lang: "en" | "hi" | "both") => {
    setSelectedLanguage(lang);
    if (lang === "en") enqueue("Language set to English.", "en", true);
    else if (lang === "hi") enqueue("भाषा हिन्दी चुनी गई।", "hi", true);
    else enqueue("Language set to both Hindi and English.", "en", true);
  };

  return (
    <main className="flex flex-col items-center justify-center min-h-screen bg-black text-green-400 p-4">
      <h1 className="text-3xl font-bold mb-4">👁 Blind Vision Assistant (Hindi & English)</h1>

      <div className="flex gap-3 mb-4">
        <button
          onClick={() => setupCamera()}
          className="bg-green-500 hover:bg-green-600 text-black px-4 py-2 rounded-xl font-semibold"
        >
          🎥 Start Camera
        </button>

        <div className="flex items-center gap-2 bg-gray-900 px-3 py-2 rounded-xl">
          <label className="text-sm">Language:</label>
          <select
            value={selectedLanguage}
            onChange={(e) => changeLanguage(e.target.value as "en" | "hi" | "both")}
            className="bg-black text-white px-2 py-1 rounded"
          >
            <option value="en">English</option>
            <option value="hi">हिन्दी</option>
            <option value="both">Both (हिन्दी + English)</option>
          </select>
        </div>
      </div>

      <video
        ref={videoRef}
        className="w-11/12 max-w-lg rounded-2xl border-2 border-green-400"
        autoPlay
        playsInline
        muted
      />

      <p className="mt-4 text-lg text-white text-center break-words max-w-xl">{status}</p>
      <p className="mt-2 text-sm text-gray-400 text-center">
        🎤 Say: “What do you see?”, “Stop speaking”, or “Start camera” — or change language from the dropdown.
      </p>
    </main>
  );
}
