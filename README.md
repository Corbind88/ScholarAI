scholarai/
  server/
    server.js
    package.json
    .env.example
    data/
      store.json           # created on first run
  client/                  # Angular app (we’ll generate it, then drop files in)
    proxy.conf.json
    (Angular files…)


To Run it

# 1) start the backend
cd scholarai/server
cp .env.example .env          # paste your key
npm run dev

# 2) start the Angular app (in another terminal)
cd ../client
ng serve --proxy-config proxy.conf.json
Frontend: http://localhost:4200
Backend: http://localhost:8787

3) Usage

Drop in PDF/DOCX/TXT notes.

Select which docs to search (or leave all).

Ask a question. You’ll get a grounded answer + simple bracket citations linking to the retrieved chunks.

Click Summarize to get quick study bullets for any doc.

4) What’s included / why it’s “easy”

Safe API key: kept on the server; browser only hits /api/*. (OpenAI chat + embeddings per docs.) 
OpenAI Platform
+1

Local JSON store: no DB setup—persists to server/data/store.json.

Simple retriever: naive cosine over chunks; adjustable k.

Extensible endpoints: /api/upload, /api/ask, /api/summarize.