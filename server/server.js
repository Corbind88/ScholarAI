import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import mammoth from 'mammoth';
import { fileURLToPath } from 'url';
import { v4 as uuidv4 } from 'uuid';
import OpenAI from 'openai';
import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf.mjs';

// Optional: silence worker requirement for Node usage
try { pdfjsLib.GlobalWorkerOptions.workerSrc = undefined; } catch {}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = process.env.PORT || 8787;
const DATA_DIR = path.join(__dirname, 'data');
const STORE_PATH = path.join(DATA_DIR, 'store.json');

// ---- RAG tuning knobs ----
const MAX_CHARS_PER_CHUNK = 3500;   // target size per chunk
const CHUNK_OVERLAP = 300;          // overlap between chunks
const MAX_CHUNKS_PER_DOC = 500;     // hard cap to avoid OOM / huge requests
const EMBEDDING_BATCH_SIZE = 64;    // batch size for embeddings.create

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// ---- Multer (memory) with limits ----
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024, files: 10 }, // 20MB each, max 10 files
});

// ---- OpenAI client ----
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ------------------ Robust store helpers ------------------
function ensureDataDir() {
  try { fs.mkdirSync(DATA_DIR, { recursive: true }); } catch {}
}

function loadStore() {
  ensureDataDir();
  try {
    const raw = fs.readFileSync(STORE_PATH, 'utf-8').trim();
    if (!raw) throw new Error('empty store');
    const parsed = JSON.parse(raw);
    if (!parsed.docs) parsed.docs = [];
    return parsed;
  } catch {
    const safe = { docs: [] };
    try { fs.writeFileSync(STORE_PATH, JSON.stringify(safe, null, 2)); } catch {}
    return safe;
  }
}

function saveStore(store) {
  ensureDataDir();
  const tmp = STORE_PATH + '.tmp';
  fs.writeFileSync(tmp, JSON.stringify(store, null, 2));
  fs.renameSync(tmp, STORE_PATH);
}

ensureDataDir();
if (!fs.existsSync(STORE_PATH)) saveStore({ docs: [] });

// ------------------ Utilities ------------------
function chunkText(text, chunkSize = MAX_CHARS_PER_CHUNK, overlap = CHUNK_OVERLAP) {
  const cleaned = (text || '')
    .replace(/\r\n/g, '\n')
    .replace(/\s+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n');

  if (!cleaned.length) return [];

  const chunks = [];
  let i = 0;

  // Guard: overlap must be smaller than chunkSize
  const safeOverlap = Math.max(0, Math.min(overlap, Math.floor(chunkSize / 2)));

  while (i < cleaned.length) {
    const end = Math.min(i + chunkSize, cleaned.length);
    const slice = cleaned.slice(i, end).trim();
    if (slice) chunks.push(slice);
    // move forward with overlap
    i = end - safeOverlap;
    if (i <= 0) i = end; // avoid stuck loops on small texts
  }
  return chunks;
}

function smartChunk(wholeText, targetMaxChunks = MAX_CHUNKS_PER_DOC) {
  // Start with defaults
  let size = MAX_CHARS_PER_CHUNK;
  let chunks = chunkText(wholeText, size, CHUNK_OVERLAP);

  // If too many chunks, grow chunk size progressively until under the cap
  while (chunks.length > targetMaxChunks && size < 20000) {
    size = Math.floor(size * 1.5);
    chunks = chunkText(wholeText, size, CHUNK_OVERLAP);
  }

  // Still too many? Truncate to the cap (last resort)
  if (chunks.length > targetMaxChunks) {
    chunks = chunks.slice(0, targetMaxChunks);
  }

  return chunks;
}


function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

function toUint8Array(buf) {
  // Accept Buffer or Uint8Array; return a plain Uint8Array (not a Buffer)
  if (buf instanceof Uint8Array && !Buffer.isBuffer(buf)) return buf;
  return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
}

async function extractText(file) {
  const mime = file.mimetype || '';
  const name = (file.originalname || '').toLowerCase();

  // ---- PDF (handle common MIME and octet-stream fallback) ----
  if (
    mime === 'application/pdf' ||
    mime === 'application/x-pdf' ||
    (mime === 'application/octet-stream' && name.endsWith('.pdf'))
  ) {
    const data = toUint8Array(file.buffer);
    const pdf = await pdfjsLib.getDocument({ data }).promise;
    let out = '';
    for (let p = 1; p <= pdf.numPages; p++) {
      const page = await pdf.getPage(p);
      const content = await page.getTextContent();
      const text = content.items.map(it => it.str ?? '').join(' ');
      out += text + '\n';
    }
    return out;
  }

  // ---- DOCX ----
  if (
    mime === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
    (mime === 'application/octet-stream' && name.endsWith('.docx'))
  ) {
    const result = await mammoth.extractRawText({ buffer: file.buffer });
    return result.value || '';
  }

  // ---- TXT / MD ----
  if (mime.startsWith('text/') || name.endsWith('.txt') || name.endsWith('.md')) {
    return file.buffer.toString('utf-8');
  }

  throw new Error(`Unsupported file type: ${mime || 'unknown'} (${name})`);
}

async function embedTexts(texts, batchSize = EMBEDDING_BATCH_SIZE) {
  if (!Array.isArray(texts) || texts.length === 0) return [];

  const out = [];
  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const res = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: batch
    });
    // OpenAI returns one embedding per input, in order
    for (const d of res.data) out.push(d.embedding);
  }
  return out;
}


// ------------------ Routes ------------------
app.get('/api/health', (_req, res) => res.json({ ok: true }));

app.get('/api/docs', (_req, res) => {
  const store = loadStore();
  const docs = store.docs.map(({ id, name, chunkCount, createdAt }) => ({ id, name, chunkCount, createdAt }));
  res.json({ docs });
});

// ---- Upload with Multer error handling ----
const uploadHandler = upload.array('files', 10);

app.post('/api/upload', (req, res) => {
  uploadHandler(req, res, async (err) => {
    if (err) {
      // Multer errors (e.g., LIMIT_FILE_SIZE) won’t crash the process
      const code = err.code || 'UPLOAD_ERROR';
      const status = code === 'LIMIT_FILE_SIZE' ? 413 : 400;
      return res.status(status).json({ error: code });
    }

    try {
      if (!process.env.OPENAI_API_KEY) {
        return res.status(400).json({ error: 'Missing OPENAI_API_KEY on server' });
      }
      const files = req.files || [];
      if (!files.length) return res.status(400).json({ error: 'No files uploaded' });

      const store = loadStore();
      const out = [];

      for (const file of files) {
        let text = '';
        try {
          text = (await extractText(file)).trim();
        } catch (e) {
          // Skip problematic files but keep the server alive
          console.error('extractText failed:', e?.message || e);
          continue;
        }
        if (!text) continue;

        const chunks = smartChunk(text);
        if (!chunks.length) continue;

        // batch embeddings to avoid giant single requests
        const embeddings = await embedTexts(chunks, EMBEDDING_BATCH_SIZE);

        // Guard against any mismatch (shouldn’t happen, but be safe)
        if (embeddings.length !== chunks.length) {
        console.error('Embedding mismatch: got', embeddings.length, 'for', chunks.length, 'chunks');
        continue;
}


        const doc = {
          id: uuidv4(),
          name: file.originalname,
          createdAt: new Date().toISOString(),
          chunkCount: chunks.length,
          chunks: chunks.map((t, i) => ({ id: i, text: t, embedding: embeddings[i] }))
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

// ---- Ask (RAG) ----
app.post('/api/ask', async (req, res) => {
  try {
    const { question, docIds, k = 6 } = req.body || {};
    if (!question || typeof question !== 'string') return res.status(400).json({ error: 'Missing question' });

    const store = loadStore();
    const scope = Array.isArray(docIds) && docIds.length
      ? store.docs.filter(d => docIds.includes(d.id))
      : store.docs;

    if (!scope.length) return res.status(400).json({ error: 'No docs available. Upload first.' });

    const qEmbed = (await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: question
    })).data[0].embedding;

    const scored = [];
    for (const doc of scope) {
      for (const ch of doc.chunks) {
        scored.push({
          docId: doc.id,
          name: doc.name,
          chunkId: ch.id,
          text: ch.text,
          score: cosine(qEmbed, ch.embedding)
        });
      }
    }
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

// ---- Summarize ----
app.post('/api/summarize', async (req, res) => {
  try {
    const { docId } = req.body || {};
    const store = loadStore();
    const doc = store.docs.find(d => d.id === docId);
    if (!doc) return res.status(404).json({ error: 'Doc not found' });

    const text = doc.chunks.map(c => c.text).join('\n\n').slice(0, 100_000);
    const completion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      temperature: 0.3,
      messages: [
        { role: 'system', content: 'Summarize clearly with bullet points and key terms.' },
        { role: 'user', content: text }
      ]
    });

    res.json({ summary: completion.choices[0]?.message?.content || '' });
  } catch (err) {
    console.error('summarize route error:', err);
    res.status(500).json({ error: String(err?.message || err) });
  }
});

// ------------------ Global error + process guards ------------------
app.use((err, _req, res, _next) => {
  console.error('global error handler:', err);
  res.status(500).json({ error: String(err?.message || err) });
});

process.on('uncaughtException', (err) => {
  console.error('uncaughtException:', err);
  // keep process alive; at worst, healthcheck fails but server continues
});
process.on('unhandledRejection', (reason) => {
  console.error('unhandledRejection:', reason);
});

app.listen(PORT, () => {
  console.log(`ScholarAI server listening on http://localhost:${PORT}`);
});
