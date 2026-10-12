/**
 * Per-chunk timing for the Zipformer engine on native onnxruntime-node.
 *
 * Feeds 300 ms chunks (the real-time deadline) and times every feed() call,
 * which includes the encoder run plus decode, search and tracker work.
 *
 *   THREADS=1 npx tsx test/native-bench.ts [clip ...]
 *
 * Reports per-chunk p50 / p95 / max and real-time factor = compute / audio.
 */
import { createRequire } from "node:module";
import { execSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { displayQuranFromRaw, ZipformerSession } from "../src/worker/zipformer-session.ts";
import type { ZipformerIo } from "@tilawa/core";

const HERE = dirname(fileURLToPath(import.meta.url));
const FRONTEND = resolve(HERE, "..");
const WORKTREE = resolve(FRONTEND, "../..");
const SAMPLE_RATE = 16000;
const CHUNK = 4800; // 300 ms
const THREADS = Number(process.env.THREADS ?? "1");

const MODEL = resolve(FRONTEND, "public/models/zipformer_a0w_ep1_a05.int8.onnx");
const IO_PATH = resolve(FRONTEND, "public/models/zipformer_a0w_ep1_a05.io.json");
const CORPUS = resolve(FRONTEND, "public/zipformer_quran.json");
const QURAN = resolve(FRONTEND, "public/quran.json");
const CORPUS_DIR = resolve(WORKTREE, "lab/benchmark/test_corpus");

const clipArgs = process.argv.slice(2);
const clips = clipArgs.length
  ? clipArgs
  : ["001002.mp3", "002255.mp3", "036001.mp3", "112001.mp3", "long_002_285.wav", "long_023_115.wav"];

function loadAudio(filePath: string): Float32Array {
  const buf = execSync(
    `ffmpeg -hide_banner -loglevel error -i "${filePath}" -f f32le -ar ${SAMPLE_RATE} -ac 1 pipe:1`,
    { maxBuffer: 200 * 1024 * 1024 },
  );
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

const require = createRequire(`${resolve(FRONTEND, "node_modules")}/`);
const ort = require("onnxruntime-node");

// Pin ORT's thread pools. The engine's runner does not expose these options,
// so inject them into every session it creates.
const origCreate = ort.InferenceSession.create.bind(ort.InferenceSession);
ort.InferenceSession.create = (model: Uint8Array, opts: Record<string, unknown> = {}) =>
  origCreate(model, { ...opts, intraOpNumThreads: THREADS, interOpNumThreads: 1 });

const io = JSON.parse(readFileSync(IO_PATH, "utf8")) as ZipformerIo;
const corpus = JSON.parse(readFileSync(CORPUS, "utf8"));
const quran = displayQuranFromRaw(JSON.parse(readFileSync(QURAN, "utf8")));
const modelBytes = new Uint8Array(readFileSync(MODEL));

function pct(sorted: number[], p: number): number {
  return sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))]!;
}

let totalAudio = 0;
let totalCompute = 0;
const allChunks: number[] = [];

for (const clip of clips) {
  const path = existsSync(clip) ? clip : resolve(CORPUS_DIR, clip);
  const pcm = loadAudio(path);
  const host = await ZipformerSession.create({
    ort,
    model: modelBytes,
    io,
    corpus,
    quran,
    executionProviders: ["cpu"],
  });
  const times: number[] = [];
  for (let i = 0; i < pcm.length; i += CHUNK) {
    const chunk = pcm.subarray(i, Math.min(pcm.length, i + CHUNK));
    const t0 = performance.now();
    await host.feed(chunk);
    times.push(performance.now() - t0);
  }
  const t0 = performance.now();
  await host.stop();
  times.push(performance.now() - t0);

  const audioS = pcm.length / SAMPLE_RATE;
  const computeS = times.reduce((a, b) => a + b, 0) / 1000;
  totalAudio += audioS;
  totalCompute += computeS;
  allChunks.push(...times);
  console.log(`${clip}\taudio=${audioS.toFixed(1)}s\tcompute=${computeS.toFixed(2)}s\trtf=${(computeS / audioS).toFixed(3)}`);
}

const sorted = [...allChunks].sort((a, b) => a - b);
console.log(
  `THREADS=${THREADS} clips=${clips.length} audio=${totalAudio.toFixed(1)}s compute=${totalCompute.toFixed(2)}s ` +
  `rtf=${(totalCompute / totalAudio).toFixed(3)} ` +
  `chunk_ms p50=${pct(sorted, 50).toFixed(1)} p95=${pct(sorted, 95).toFixed(1)} max=${sorted.at(-1)!.toFixed(1)} ` +
  `(deadline 300 ms)`,
);
