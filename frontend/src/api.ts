// Central API base URL for the frontend. Use Vite env var `VITE_API` when provided,
// otherwise default to local backend used during development.
export const API = (import.meta as any).env?.VITE_API || 'http://localhost:8000';
export default API;

const jsonHeaders = {
  'Content-Type': 'application/json',
};

async function request<T>(path: string, options: RequestInit = {}) {
  const response = await fetch(`${API}${path}`, {
    credentials: 'same-origin',
    ...options,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`API request failed: ${response.status} ${response.statusText} - ${text}`);
  }

  // Some endpoints may return empty body
  const txt = await response.text();
  return txt ? (JSON.parse(txt) as T) : ({} as T);
}

export async function fetchGraph() {
  return request<{ nodes: any[]; edges: any[] }>('/api/signals/graph');
}

export async function generateReport(ticker: string, company: string, mode: string = 'full') {
  return request<any>('/api/reports/generate', {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify({ ticker, company, mode }),
  });
}

export async function chat(messages: { role: 'user' | 'assistant' | 'system'; content: string }[] | any, ticker?: string) {
  return request<{ content: string } | any>('/api/chat/', {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify({ messages, ticker }),
  });
}

export function createEventSource(path: string, onMessage: (data: any) => void) {
  const source = new EventSource(`${API}${path}`);
  source.onmessage = (event) => {
    try {
      onMessage(JSON.parse(event.data));
    } catch (error) {
      console.warn('Unable to parse SSE event data', error);
    }
  };
  source.onerror = (err) => console.warn('SSE error', err);
  return source;
}

export async function triggerTicker(ticker: string) {
  return request<any>(`/api/signals/trigger/${encodeURIComponent(ticker)}`, {
    method: 'POST',
  });
}
