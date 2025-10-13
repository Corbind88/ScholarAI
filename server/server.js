import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import os from 'os';
import mammoth from 'mammoth';
import { fileURLToPath } from 'url';
import { v4 as uuidv4 } from 'uuid';
import OpenAI from 'openai';
import { spawn, execFileSync } from 'child_process';

// ---------- Paths / constants ----------
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = process.env.PORT || 8787;
const DATA_DIR = path.join(__dirname, 'data');
const STORE_PATH = path.join(DATA_DIR, 'store.json');

// RAG knobs
const MAX_CHARS_PER_CHUNK  = 3000;     // slightly smaller chunks -> lower peak memory
const CHUNK_OVERLAP        = 200;
const MAX_CHUNKS_PER_DOC   = 300;      // streamed cap per uploaded doc
const MAX_CHUNKS_FOR_ASK   = 800;      // total per /api/ask across docs
const EMBEDDING_BATCH_SIZE = 64;

// Extraction caps (strict)
const MAX_TEXT_PER_FILE    = 80_000;   // we won't read more than this many chars from any file
const MAX_STORE_BYTES      = 5 * 1024 * 1024; // refuse >5MB store.json (auto-reset)
const PDF_TIMEOUT_MS       = 8000;     // kill pdftotext after 8s (safety)
const STREAM_HWM           = 16 * 1024; // 16KB stream buffer

// ---------- App ----------
const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Multer: disk storage (no Buffers held in memory)
const UPLOAD_DIR = path.join(os.tmpdir(), 'scholarai-uploads');
fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, UPLOAD_DIR),
  filename: (_req, file, cb) => {
    const safe = (file.originalname || 'upload').replace(/[^\w.\-]+/g, '_');
    cb(null, `${Date.now()}-${Math.random().toString(36).slice(2)}-${safe}`);
  }
});
const upload = multer({
  storage,
  limits: { fileSize: 20 * 1024 * 1024, files: 10 }, // 20MB/file
});

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Detect Poppler
let PDFTOTEXT_AVAILABLE = false;
try { execFileSync('pdftotext', ['-v'], { stdio: 'ignore' }); PDFTOTEXT_AVAILABLE = true; } catch {}
console.log('pdftotext available:', PDFTOTEXT_AVAILABLE);

// ---------- Store (self-heal, text-only) ----------
function ensureDataDir() { try { fs.mkdirSync(DATA_DIR, { recursive: true }); } catch {} }
function writeStore(obj) {
  const tmp = STORE_PATH + '.tmp';
  fs.writeFileSync(tmp, JSON.stringify(obj, null, 2));
  fs.renameSync(tmp, STORE_PATH);
}
function loadStore() {
  ensureDataDir();
  try {
    if (!fs.existsSync(STORE_PATH)) { const s = { docs: [] }; writeStore(s); return s; }
    const st = fs.statSync(STORE_PATH);
    if (st.size === 0 || st.size > MAX_STORE_BYTES) { const s = { docs: [] }; writeStore(s); return s; }
    const raw = fs.readFileSync(STORE_PATH, 'utf-8').trim();
    if (!raw) { const s = { docs: [] }; writeStore(s); return s; }
    const parsed = JSON.parse(raw);
    if (!parsed.docs) parsed.docs = [];
    // strip legacy embeddings
    for (const d of parsed.docs) if (Array.isArray(d?.chunks)) for (const c of d.chunks) if (c && 'embedding' in c) delete c.embedding;
    return parsed;
  } catch {
    const s = { docs: [] }; writeStore(s); return s;
  }
}
function saveStore(store) { ensureDataDir(); writeStore(store); }

// ---------- Chunking (stream-friendly) ----------
function pushChunksFromBuffer(chunks, buffer) {
  while (buffer.value.length >= MAX_CHARS_PER_CHUNK && chunks.length < MAX_CHUNKS_PER_DOC) {
    const slice = buffer.value.slice(0, MAX_CHARS_PER_CHUNK).trim();
    if (slice) chunks.push(slice);
    buffer.value = buffer.value.slice(MAX_CHARS_PER_CHUNK - CHUNK_OVERLAP);
  }
}

async function streamReadableToChunks(rs) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    const buf = { value: '' };
    let total = 0;
    let ended = false;

    rs.setEncoding('utf8');
    rs.on('data', (chunk) => {
      if (ended) return;
      // enforce total char cap
      const remain = MAX_TEXT_PER_FILE - total;
      if (remain <= 0 || chunks.length >= MAX_CHUNKS_PER_DOC) {
        ended = true;
        try { rs.destroy(); } catch {}
        return;
      }
      // normalize small piece only
      let piece = chunk.replace(/\r\n/g, '\n');
      if (piece.length > remain) piece = piece.slice(0, remain);
      total += piece.length;

      buf.value += piece;
      pushChunksFromBuffer(chunks, buf);

      if (chunks.length >= MAX_CHUNKS_PER_DOC || total >= MAX_TEXT_PER_FILE) {
        ended = true;
        try { rs.destroy(); } catch {}
      }
    });
    rs.on('error', (e) => reject(e));
    rs.on('close', () => {
      if (!ended) {
        // flush remainder as a final chunk
        const tail = buf.value.trim();
        if (tail && chunks.length < MAX_CHUNKS_PER_DOC) chunks.push(tail);
      }
      resolve(chunks);
    });
    rs.on('end', () => {
      // handled by 'close', but keep for safety
    });
  });
}

// ---------- Extractors (streaming; no big strings) ----------
function extractChunksFromTxtPath(p) {
  const rs = fs.createReadStream(p, { encoding: 'utf8', highWaterMark: STREAM_HWM });
  return streamReadableToChunks(rs);
}
async function extractChunksFromDocxPath(p) {
  // DOCX -> mammoth returns a single string; cap aggressively, then chunk
  const res = await mammoth.extractRawText({ path: p });
  const text = (res.value || '').slice(0, MAX_TEXT_PER_FILE);
  const chunks = [];
  const buf = { value: '' };
  let i = 0;
  while (i < text.length && chunks.length < MAX_CHUNKS_PER_DOC) {
    const take = Math.min(MAX_CHARS_PER_CHUNK, text.length - i);
    buf.value += text.slice(i, i + take);
    i += take;
    pushChunksFromBuffer(chunks, buf);
  }
  const tail = buf.value.trim();
  if (tail && chunks.length < MAX_CHUNKS_PER_DOC) chunks.push(tail);
  return chunks;
}
function extractChunksFromPdfPath(p) {
  if (!PDFTOTEXT_AVAILABLE) {
    const err = new Error('PDFTOTEXT_MISSING');
    err.code = 'PDFTOTEXT_MISSING';
    throw err;
  }
  return new Promise((resolve, reject) => {
    const child = spawn('pdftotext', ['-layout', '-nopgbrk', '-enc', 'UTF-8', p, '-'], { stdio: ['ignore', 'pipe', 'ignore'] });
    const rs = child.stdout;
    let timer = setTimeout(() => { try { child.kill('SIGKILL'); } catch {} }, PDF_TIMEOUT_MS);

    streamReadableToChunks(rs)
      .then((chunks) => {
        clearTimeout(timer);
        try { child.kill('SIGTERM'); } catch {}
        resolve(chunks);
      })
      .catch((e) => {
        clearTimeout(timer);
        try { child.kill('SIGKILL'); } catch {}
        reject(e);
      });

    child.on('error', (e) => {
      clearTimeout(timer);
      reject(e);
    });
  });
}

async function extractFileToChunks(filePath, mimeGuess, origLower) {
  if (origLower.endsWith('.pdf') || mimeGuess === 'application/pdf' || mimeGuess === 'application/x-pdf') {
    return extractChunksFromPdfPath(filePath);
  }
  if (origLower.endsWith('.docx') || mimeGuess === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
    return extractChunksFromDocxPath(filePath);
  }
  // TXT / MD
  if (origLower.endsWith('.txt') || origLower.endsWith('.md') || (mimeGuess || '').startsWith('text/')) {
    return extractChunksFromTxtPath(filePath);
  }
  throw new Error(`Unsupported file type: ${mimeGuess || 'unknown'} (${origLower})`);
}

// ---------- Embeddings (on-demand, batched) ----------
async function embedTexts(texts, batchSize = EMBEDDING_BATCH_SIZE) {
  if (!Array.isArray(texts) || !texts.length) return [];
  const out = [];
  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const res = await openai.embeddings.create({ model: 'text-embedding-3-small', input: batch });
    for (const d of res.data) out.push(d.embedding);
  }
  return out;
}
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}
function limitScopeChunks(scopeDocs, cap = MAX_CHUNKS_FOR_ASK) {
  const total = scopeDocs.reduce((s, d) => s + d.chunks.length, 0);
  if (total <= cap) return scopeDocs.map(d => ({ doc: d, chunks: d.chunks }));
  const perDoc = Math.max(1, Math.floor(cap / scopeDocs.length));
  return scopeDocs.map(d => ({ doc: d, chunks: d.chunks.slice(0, perDoc) }));
}

// ---------- Routes ----------
app.get('/api/health', (_req, res) => res.json({ ok: true }));

app.get('/api/docs', (_req, res) => {
  const store = loadStore();
  const docs = store.docs.map(({ id, name, chunkCount, createdAt }) => ({ id, name, chunkCount, createdAt }));
  res.json({ docs });
});

// Upload: stream-extract -> store ONLY TEXT
const uploadHandler = upload.array('files', 10);

app.post('/api/upload', (req, res) => {
  uploadHandler(req, res, async (err) => {
    if (err) {
      const code = err.code || 'UPLOAD_ERROR';
      const status = code === 'LIMIT_FILE_SIZE' ? 413 : 400;
      return res.status(status).json({ error: code });
    }

    try {
      if (!process.env.OPENAI_API_KEY) return res.status(400).json({ error: 'Missing OPENAI_API_KEY on server' });
      const files = req.files || [];
      if (!files.length) return res.status(400).json({ error: 'No files uploaded' });

      const store = loadStore();
      const out = [];

      for (const file of files) {
        const nameLower = (file.originalname || '').toLowerCase();
        let chunks = [];
        try {
          chunks = await extractFileToChunks(file.path, file.mimetype || '', nameLower);
        } catch (e) {
          if (e?.code === 'PDFTOTEXT_MISSING') {
            return res.status(501).json({
              error: 'PDFTOTEXT_MISSING',
              fix: 'Install Poppler (pdftotext) and restart. On macOS: brew install poppler'
            });
          }
          console.error('extractFileToChunks failed:', e?.message || e);
          // continue to next file
          continue;
        } finally {
          // always remove uploaded temp file
          try { fs.unlinkSync(file.path); } catch {}
        }

        if (!chunks.length) continue;

        const doc = {
          id: uuidv4(),
          name: file.originalname,
          createdAt: new Date().toISOString(),
          chunkCount: chunks.length,
          chunks: chunks.map((t, i) => ({ id: i, text: t })) // store only text
        };

        store.docs.push(doc);
        out.push({ id: doc.id, name: doc.name, chunkCount: doc.chunkCount });
      }

      saveStore(store);
      res.json({ uploaded: out });
    } catch (e) {
      console.error('upload route error:', e);
      res.status(500).json({ error: String(e?.message || e) });
    }
  });
});

// Ask: embed question + (limited) chunks on the fly; then rank
app.post('/api/ask', async (req, res) => {
  try {
    const { question, docIds, k = 6 } = req.body || {};
    if (!question || typeof question !== 'string') return res.status(400).json({ error: 'Missing question' });

    const store = loadStore();
    const scopeDocs = (Array.isArray(docIds) && docIds.length)
      ? store.docs.filter(d => docIds.includes(d.id))
      : store.docs;

    if (!scopeDocs.length) return res.status(400).json({ error: 'No docs available. Upload first.' });

    const packs = limitScopeChunks(scopeDocs, MAX_CHUNKS_FOR_ASK);
    const all = [];
    for (const { doc, chunks } of packs) for (const ch of chunks) all.push({ docId: doc.id, name: doc.name, chunkId: ch.id, text: ch.text });

    const qEmbed = (await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: question
    })).data[0].embedding;

    const chunkEmbeds = await embedTexts(all.map(x => x.text), EMBEDDING_BATCH_SIZE);

    const scored = all.map((item, i) => ({ ...item, score: cosine(qEmbed, chunkEmbeds[i]) }));
    scored.sort((a, b) => b.score - a.score);
    const top = scored.slice(0, k);

    const contextBlocks = top.map((t, idx) =>
      `--- Source ${idx + 1} | ${t.name} | chunk ${t.chunkId} | score ${t.score.toFixed(3)} ---\n${t.text}`
    ).join('\n\n');

    const system = `You are ScholarAI, a study assistant. Answer the user using ONLY the provided sources.
If the answer isn’t in the sources, say you don’t have enough information.
Cite like [${top.map((_, i) => i + 1).join(', ')}] where relevant. Be concise and helpful.`;

    const completion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      temperature: 0.2,
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: `QUESTION:\n${question}\n\nSOURCES:\n${contextBlocks}` }
      ]
    });

    res.json({
      answer: completion.choices[0]?.message?.content || '',
      citations: top.map((t, i) => ({ label: i + 1, name: t.name, chunkId: t.chunkId, docId: t.docId, score: t.score }))
    });
  } catch (err) {
    console.error('ask route error:', err);
    res.status(500).json({ error: String(err?.message || err) });
  }
});

// Global guards
app.use((err, _req, res, _next) => {
  console.error('global error handler:', err);
  res.status(500).json({ error: String(err?.message || err) });
});
process.on('uncaughtException', (err) => console.error('uncaughtException:', err));
process.on('unhandledRejection', (reason) => console.error('unhandledRejection:', reason));

app.listen(PORT, () => console.log(`ScholarAI server listening on http://localhost:${PORT}`));
