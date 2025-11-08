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

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [status, setStatus] = useState("Waiting for camera permission...");
  const [isStarted, setIsStarted] = useState(false);
  const [voice, setVoice] = useState<SpeechSynthesisVoice | null>(null);
  const [lastSpoken, setLastSpoken] = useState("");
  const detectingRef = useRef(false);
  const recentLabels = useRef<string[]>([]);
  const lastAnnouncementTime = useRef(0);
  const MIN_ANNOUNCEMENT_INTERVAL = 3000;
  const frameSkipCounter = useRef(0);

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
        speak("Camera activated. I will announce objects and people I detect.");
      }
    } catch (err) {
      console.error("Camera error:", err);
      setStatus("Camera permission denied or unavailable.");
    }
  };

  useEffect(() => {
    const loadVoice = () => {
      const voices = window.speechSynthesis.getVoices();
      if (voices.length > 0) {
        const best =
          voices.find((v) => v.name?.includes("Google US English")) ||
          voices.find((v) => v.lang?.startsWith("en")) ||
          voices[0];
        setVoice(best || null);
      }
    };
    loadVoice();
    window.speechSynthesis.onvoiceschanged = loadVoice;
  }, []);

  const speak = (text: string) => {
    if (!("speechSynthesis" in window) || !text) return;
    if (text === lastSpoken) return;
    const now = Date.now();
    if (now - lastAnnouncementTime.current < MIN_ANNOUNCEMENT_INTERVAL) return;
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 1.0;
    utter.pitch = 1.0;
    if (voice) utter.voice = voice;
    setLastSpoken(text);
    lastAnnouncementTime.current = now;
    window.speechSynthesis.speak(utter);
  };

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

  useEffect(() => {
    if (!model || !isStarted) return;

    const detect = async () => {
      if (!videoRef.current || detectingRef.current || videoRef.current.readyState < 2) {
        requestAnimationFrame(detect);
        return;
      }

      frameSkipCounter.current++;
      if (frameSkipCounter.current % 2 !== 0) {
        requestAnimationFrame(detect);
        return;
      }

      detectingRef.current = true;
      const predictions = (await model.detect(videoRef.current)) as DetectedObject[];
      detectingRef.current = false;

      let label = "nothing";
      let currentPred: DetectedObject | null = null;
      if (predictions.length > 0) {
        const sorted = predictions.sort((a, b) => b.score - a.score);
        const top = sorted.find((p) => p.score > 0.4);
        if (top) {
          currentPred = top;
          label = top.class;
          if (label === "person") label = "person";
        }
      }

      recentLabels.current.push(label);
      if (recentLabels.current.length > 8) recentLabels.current.shift();

      const stable = getStableLabel(recentLabels.current);
      if (stable !== "nothing") {
        const msg =
          stable === "person" ? "A person is near you." : `Near you is a ${stable}.`;
        setStatus(msg);
        speak(msg);
      } else {
        setStatus("Nothing detected");
      }

      requestAnimationFrame(detect);
    };

    detect();
  }, [model, isStarted, voice]);

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
      if (transcript.includes("what do you see")) {
        // Basic query: speak current status
        speak(status);
      } else if (transcript.includes("stop speaking")) {
        window.speechSynthesis.cancel();
        lastAnnouncementTime.current = Date.now();
      } else if (transcript.includes("start camera")) {
        setupCamera();
      }
    };

    recognition.onerror = (err: any) => console.error("Speech recognition error:", err);
    recognition.onend = () => recognition.start();
    recognition.start();

    return () => recognition.stop();
  }, [status, voice]);

  const getStableLabel = (arr: string[]) => {
    const counts: Record<string, number> = {};
    for (const val of arr) counts[val] = (counts[val] || 0) + 1;
    const maxCount = Math.max(...Object.values(counts));
    const majorityThreshold = arr.length * 0.5;
    if (maxCount < majorityThreshold) return "nothing";
    return Object.keys(counts).reduce((a, b) => (counts[a] > counts[b] ? a : b));
  };

  return (
    <main className="flex flex-col items-center justify-center min-h-screen bg-black text-green-400 p-4">
      <h1 className="text-3xl font-bold mb-4">👁 Blind Vision Assistant</h1>

      {!isStarted && (
        <button
          onClick={setupCamera}
          className="bg-green-500 hover:bg-green-600 text-black px-6 py-3 rounded-xl font-semibold mb-4"
        >
          🎥 Start Camera
        </button>
      )}

      <video
        ref={videoRef}
        className="w-11/12 max-w-lg rounded-2xl border-2 border-green-400"
        autoPlay
        playsInline
        muted
      />

      <p className="mt-4 text-lg text-white text-center">{status}</p>
      <p className="mt-2 text-sm text-gray-400 text-center">
        🎤 Say: “What do you see?”, “Stop speaking”, or “Start camera”
      </p>
    </main>
  );
}
