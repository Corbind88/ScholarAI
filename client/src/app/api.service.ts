import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

export type DocMeta = { id: string; name: string; chunkCount: number; createdAt: string };

@Injectable({ providedIn: 'root' })
export class ApiService {
  constructor(private http: HttpClient) {}

  health() {
    return this.http.get<{ ok: boolean }>('/api/health');
  }

  listDocs() {
    return this.http.get<{ docs: DocMeta[] }>('/api/docs');
  }

  upload(files: File[]) {
    const form = new FormData();
    files.forEach(f => form.append('files', f, f.name));
    return this.http.post<{ uploaded: DocMeta[] }>('/api/upload', form);
  }

  ask(question: string, docIds: string[], k = 6) {
    return this.http.post<{ answer: string, citations: any[] }>('/api/ask', { question, docIds, k });
  }

  summarize(docId: string) {
    return this.http.post<{ summary: string }>('/api/summarize', { docId });
  }
}
