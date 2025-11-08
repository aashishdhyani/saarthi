// global.d.ts
interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
}

interface Window {
  webkitSpeechRecognition?: any;
}

