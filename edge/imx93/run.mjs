// CPU-only speech-to-verse runner for the i.MX93 (guest or board).
// Uses the shipped @tilawa/core SDK build and onnxruntime-node on the CPU.
//   node run.mjs <audio.f32>      raw float32 mono 16 kHz PCM
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire("/opt/tilawa/");
const ort = require("onnxruntime-node");
const { createZipformerSession, displayQuranFromRaw } = await import("/opt/tilawa/core/dist/index.js");

const DATA = "/opt/tilawa/data";
const CHUNK = 4800; // 300 ms at 16 kHz: the real-time deadline

const [, , audioPath] = process.argv;
if (!audioPath) {
  console.error("usage: node run.mjs <audio.f32>");
  process.exit(2);
}

const t0 = performance.now();
const session = await createZipformerSession({
  ort,
  model: new Uint8Array(readFileSync(`${DATA}/model.onnx`)),
  io: JSON.parse(readFileSync(`${DATA}/model.io.json`, "utf8")),
  corpus: JSON.parse(readFileSync(`${DATA}/corpus.json`, "utf8")),
  quran: displayQuranFromRaw(JSON.parse(readFileSync(`${DATA}/quran.json`, "utf8"))),
  executionProviders: ["cpu"],
});
console.log(`ready in ${(performance.now() - t0).toFixed(0)} ms`);

const raw = readFileSync(audioPath);
const ab = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
const pcm = new Float32Array(ab);
console.log(`audio: ${(pcm.length / 16000).toFixed(2)} s`);

const events = [];
const times = [];
for (let i = 0; i < pcm.length; i += CHUNK) {
  const chunk = pcm.subarray(i, Math.min(pcm.length, i + CHUNK));
  const t = performance.now();
  events.push(...(await session.feed(chunk)));
  times.push(performance.now() - t);
}
const tStop = performance.now();
events.push(...(await session.stop()));
times.push(performance.now() - tStop);

const matches = events.filter((e) => e.type === "verse_match").map((e) => `${e.surah}:${e.ayah}`);
const final = events.filter((e) => e.type === "final_sequence").at(-1);
const finalRefs = final && final.verses ? final.verses.map((v) => `${v.surah}:${v.ayah}`) : [];

const sorted = [...times].sort((a, b) => a - b);
const pct = (p) => sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))];
const computeS = times.reduce((a, b) => a + b, 0) / 1000;
console.log(`verse_match: ${matches.join(", ") || "(none)"}`);
console.log(`final:       ${finalRefs.join(", ") || "(none)"}`);
console.log(`chunks=${times.length} p50=${pct(50).toFixed(1)}ms p95=${pct(95).toFixed(1)}ms max=${sorted.at(-1).toFixed(1)}ms`);
console.log(`compute=${computeS.toFixed(2)} s  rtf=${(computeS / (pcm.length / 16000)).toFixed(3)}`);
