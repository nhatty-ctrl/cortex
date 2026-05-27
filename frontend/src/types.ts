export interface GraphNode {
  id: string;
  label: string;
  group: 'Data' | 'Analysis' | 'Synthesis' | 'Output' | 'Ticker';
  conviction?: number;
  signalDirection?: 'bullish' | 'bearish' | 'hold' | 'neutral';
  isFiring?: boolean;
}

export interface GraphEdge {
  source: string;
  target: string;
  impactMagnitude: number;
}

export interface Report {
  id: string;
  title: string;
  subtitle: string;
  format: string;
  date: string;
  content: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  report?: Report;
}
