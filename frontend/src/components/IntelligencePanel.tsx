import { useState, useRef, useEffect } from 'react';
import { Report, GraphNode } from '../types';
import { ArrowUp, AlertTriangle, TrendingUp, Globe, X, BarChart2, Clock } from 'lucide-react';
import { chat, createEventSource } from '../api';
import NodeDetails from './NodeDetails';
import ReportViewer from './ReportViewer';

interface IntelligencePanelProps {
  selectedContext: GraphNode | Report | null;
  onClearContext: () => void;
  isPanelVisible?: boolean;
  viewMode?: 'network' | 'dashboard';
}

interface ChatMsg {
  id: string;
  isUser: boolean;
  content?: string;
  type?: 'crash' | 'alpha' | 'geo' | 'compare' | 'asset' | 'alert_critical' | 'alert_high' | 'alert_watch' | 'thinking';
  assetName?: string;
  modeName?: string;
}

export default function IntelligencePanel({ selectedContext, onClearContext, isPanelVisible, viewMode = 'network' }: IntelligencePanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [messages, setMessages] = useState<ChatMsg[]>([
    { id: '2', isUser: false, type: 'alert_critical' },
  ]);
  const [isChatOpen, setIsChatOpen] = useState(false);

  const [activeAssets, setActiveAssets] = useState<Set<string>>(new Set(['NVDA','AAPL','GOLD','TSLA','EURUSD']));
  const [activeMode, setActiveMode] = useState<string>('full report');
  const [inputValue, setInputValue] = useState('');
  const [scrapeLogs, setScrapeLogs] = useState<Array<{ timestamp: string; message: string; source?: string; status?: string }>>([]);
  const [isScrapeLive, setIsScrapeLive] = useState(false);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isChatOpen]);

  useEffect(() => {
    const source = createEventSource('/api/signals/live-scrape', (event) => {
      if (!event) return;
      if (event.type === 'scrape_status' || event.type === 'agent_progress') {
        setScrapeLogs(prev => [
          {
            timestamp: event.timestamp || new Date().toISOString(),
            message: event.message || JSON.stringify(event),
            source: event.source,
            status: event.status,
          },
          ...prev,
        ].slice(0, 8));
      }
    });

    source.onopen = () => setIsScrapeLive(true);
    source.onerror = () => setIsScrapeLive(false);

    return () => {
      source.close();
    };
  }, []);

  const toggleAsset = (val: string) => {
    setActiveAssets(prev => {
      const next = new Set(prev);
      if (next.has(val)) next.delete(val);
      else next.add(val);
      return next;
    });
  };

  const toggleMode = (val: string) => {
    setActiveMode(val);
  };

  const addCustomAsset = () => {
    const ticker = prompt('Enter ticker or asset name:');
    if (!ticker) return;
    const t = ticker.toUpperCase().trim();
    setActiveAssets(prev => new Set(prev).add(t));
  };

  const fillQ = (text: string) => {
    setInputValue(text);
    setIsChatOpen(true);
  };

  const handleSend = async () => {
    const q = inputValue.trim();
    if (!q) return;

    setIsChatOpen(true);
    setInputValue('');

    const thinkingId = `${Date.now()}-thinking`;
    setMessages(prev => [...prev, { id: thinkingId, isUser: false, type: 'thinking' }]);

    const payloadMessages = [
      ...messages
        .filter(m => m.content && m.type !== 'thinking')
        .map(m => ({ role: m.isUser ? 'user' : 'assistant', content: m.content! })),
      { role: 'user' as const, content: q }
    ];

    let ticker: string;
    if (selectedContext && 'group' in selectedContext && (selectedContext as GraphNode).group === 'Ticker') {
      ticker = (selectedContext as GraphNode).label;
    } else {
      const first = Array.from(activeAssets)[0] as string | undefined;
      ticker = first || 'NVDA';
    }

    try {
      const response = await chat(payloadMessages, ticker);
      setMessages(prev => prev.filter(m => m.id !== thinkingId).concat({
        id: Date.now().toString(),
        isUser: false,
        content: response.content,
      }));
    } catch (error) {
      setMessages(prev => prev.filter(m => m.id !== thinkingId).concat({
        id: Date.now().toString(),
        isUser: false,
        content: 'Unable to reach the backend chat service. Please check http://localhost:8000 and try again.',
      }));
      console.error('Backend chat request failed', error);
    }
  };

  const placeholder = [...activeAssets].join(', ')
    ? `Ask about ${[...activeAssets].join(', ')} - ${activeMode}...`
    : 'Ask about any asset...';

  return (
    <div className="flex w-full h-full relative">
      <div className="flex flex-col flex-1 bg-[#0d0d0d] rounded-2xl border border-[#1e1e1e] overflow-hidden font-sans h-full">
        {/* Top Bar */}
        <div className="px-4 py-3.5 border-b border-[#1e1e1e] shrink-0">
          <div className="flex items-center justify-between mb-2.5">
            <span className="text-[12px] font-medium tracking-[0.08em] uppercase text-white">cortex</span>
            <span className="flex items-center">
              <span className="w-[7px] h-[7px] rounded-full bg-[#1D9E75] inline-block mr-1.5 animate-pulse"></span>
              <span className="text-[11px] text-[#4a4a4a]">live — scraped 2 min ago</span>
            </span>
          </div>
          <div className="grid grid-cols-3 gap-1.5">
            <div className="bg-[#161616] rounded-lg px-2.5 py-1.5">
              <p className="text-[9px] text-[#444] uppercase tracking-[0.05em] mb-1">regime</p>
              <p className="text-xs font-medium text-[#EF9F27]">Late Cycle</p>
              <p className="text-[9px] text-[#444] mt-0.5">Fed: pause</p>
            </div>
            <div className="bg-[#161616] rounded-lg px-2.5 py-1.5">
              <p className="text-[9px] text-[#444] uppercase tracking-[0.05em] mb-1">signals</p>
              <p className="text-xs font-medium"><span className="text-[#1D9E75]">4 buy</span> <span className="text-[#EF9F27]">2 hold</span></p>
              <p className="text-[9px] text-[#444] mt-0.5">1 crash warn</p>
            </div>
            <div className="bg-[#161616] rounded-lg px-2.5 py-1.5">
              <p className="text-[9px] text-[#444] uppercase tracking-[0.05em] mb-1">VIX</p>
              <p className="text-xs font-medium text-white">18.4</p>
              <p className="text-[9px] text-[#1D9E75] mt-0.5">clear</p>
            </div>
          </div>
        </div>

        {/* Scrollable Content Area */}
        <div className="flex-1 overflow-y-auto custom-scrollbar flex flex-col">
          {viewMode === 'dashboard' ? (
            <div className="flex-1 flex flex-col p-4 gap-4">
               {/* 5 min scrape reports */}
               <div className="bg-[#111] border border-[#1e1e1e] rounded-xl p-4 flex flex-col">
                 <h3 className="text-[10px] uppercase tracking-widest text-[#888] flex items-center gap-2 mb-3">
                   <Clock className="w-3 h-3 text-[#1D9E75]" />
                   Live Scrape Reports (5m)
                 </h3>
                 <div className="space-y-2">
                   {(scrapeLogs.length ? scrapeLogs : [
                     { timestamp: new Date(Date.now() - 120000).toISOString(), message: 'Unusual options volume detected in AAPL 155c', source: 'scheduler' },
                     { timestamp: new Date(Date.now() - 420000).toISOString(), message: 'Macro: Consumer sentiment index beat expectations', source: 'scheduler' },
                     { timestamp: new Date(Date.now() - 720000).toISOString(), message: 'NVDA block trade: 1.2M shares at $130', source: 'scheduler' },
                   ]).map((log, i) => (
                     <div key={i} className="flex gap-3 items-start p-2 rounded-lg bg-[#161616] border border-[#2a2a2a]">
                       <span className="text-[10px] text-[#555] shrink-0 mt-0.5 w-16">{new Date(log.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                       <span className="text-xs text-[#ccc]">{log.message}</span>
                     </div>
                   ))}
                 </div>
               </div>
               
               {/* Calculations & Analysis */}
               <div className="bg-[#111] border border-[#1e1e1e] rounded-xl p-4 flex flex-col flex-1">
                 <h3 className="text-[10px] uppercase tracking-widest text-[#888] flex items-center gap-2 mb-4">
                   <BarChart2 className="w-3 h-3 text-[#EF9F27]" />
                   Top 10 Technical Metrics & Watch
                 </h3>
                 <div className="flex items-center justify-between mb-4 px-2">
                   <div className="flex flex-col text-center">
                     <span className="text-3xl font-light text-[#1D9E75]">72%</span>
                     <span className="text-[10px] text-[#666] uppercase tracking-widest mt-1">Bull Watch</span>
                   </div>
                   <div className="w-px h-12 bg-[#2a2a2a]"></div>
                   <div className="flex flex-col text-center">
                     <span className="text-3xl font-light text-[#E24B4A]">28%</span>
                     <span className="text-[10px] text-[#666] uppercase tracking-widest mt-1">Bear Risk</span>
                   </div>
                 </div>
                 
                 <div className="space-y-4">
                   <div className="flex items-center gap-2 text-xs">
                     <span className="text-[#888] w-12 shrink-0">RSI 14</span>
                     <div className="flex-1 h-1.5 bg-[#222] rounded-full overflow-hidden">
                       <div className="h-full bg-[#1D9E75] w-[64%]"></div>
                     </div>
                     <span className="text-white w-8 text-right">64.2</span>
                   </div>
                   <div className="flex items-center gap-2 text-xs">
                     <span className="text-[#888] w-12 shrink-0">MACD</span>
                     <div className="flex-1 h-1.5 bg-[#222] rounded-full overflow-hidden flex justify-center">
                       <div className="h-full bg-[#1D9E75] w-[70%] origin-left"></div>
                     </div>
                     <span className="text-white w-8 text-right">+1.2</span>
                   </div>
                   <div className="flex items-center gap-2 text-xs">
                     <span className="text-[#888] w-12 shrink-0">VWAP</span>
                     <div className="flex-1 h-1.5 bg-[#222] rounded-full overflow-hidden">
                       <div className="h-full bg-[#EF9F27] w-[50%]"></div>
                     </div>
                     <span className="text-white w-8 text-right">Neut</span>
                   </div>
                   <div className="flex items-center gap-2 text-xs">
                     <span className="text-[#888] w-12 shrink-0">BBand</span>
                     <div className="flex-1 h-1.5 bg-[#222] rounded-full overflow-hidden">
                       <div className="h-full bg-[#E24B4A] w-[85%]"></div>
                     </div>
                     <span className="text-white w-8 text-right">High</span>
                   </div>
                 </div>
               </div>
            </div>
          ) : (
            <div className="p-4 flex flex-col items-center justify-center relative flex-1 min-h-[400px]">
              {selectedContext ? (
                'group' in selectedContext ? (
                  <NodeDetails node={selectedContext as GraphNode} onClose={onClearContext} />
                ) : (
                  <ReportViewer report={selectedContext as Report} onClose={onClearContext} />
                )
              ) : (
                <div className="flex-1 w-full flex flex-col p-4 bg-[#111] border border-[#1e1e1e] rounded-xl m-2 overflow-hidden">
                  <h3 className="text-[10px] uppercase tracking-widest text-[#1D9E75] flex items-center gap-2 mb-4 shrink-0 font-bold">
                    <Globe className="w-3 h-3" />
                    Consciousness Stream
                  </h3>
                  <div className="flex-1 overflow-y-auto custom-scrollbar space-y-3 relative pr-2">
                    <div className="animate-[pulse_4s_ease-in-out_infinite] flex gap-3 text-xs">
                       <span className="text-[#1D9E75] shrink-0 font-mono text-[10px] mt-0.5 w-[50px]">L I V E</span>
                       <span className="text-[#eee] font-medium">Scraper agent digesting 142 new SEC filings concurrently across indices...</span>
                    </div>
                    <div className="flex gap-3 text-xs opacity-90">
                       <span className="text-[#666] shrink-0 font-mono text-[10px] mt-0.5 w-[50px]">32s AGO</span>
                       <span className="text-[#ccc]">Unusual continuous block buys in MSFT ($4.2M total) detected across 3 dark pools.</span>
                    </div>
                    <div className="flex gap-3 text-xs opacity-80">
                       <span className="text-[#666] shrink-0 font-mono text-[10px] mt-0.5 w-[50px]">1m AGO</span>
                       <span className="text-[#ccc]">TSLA autopilot V12 deployment timeline accelerated. DeepSeek rating updated.</span>
                    </div>
                    <div className="flex gap-3 text-xs opacity-75">
                       <span className="text-[#666] shrink-0 font-mono text-[10px] mt-0.5 w-[50px]">2m AGO</span>
                       <span className="text-[#ccc]">Macro Tracker: Eurozone inflation preliminary data leaks suggest 0.2% beat. EUR/USD repricing.</span>
                    </div>
                    <div className="flex gap-3 text-xs opacity-60">
                       <span className="text-[#666] shrink-0 font-mono text-[10px] mt-0.5 w-[50px]">5m AGO</span>
                       <span className="text-[#ccc]">Gold spikes $12/oz on unconfirmed reports of new PBOC accumulation.</span>
                    </div>
                    <div className="flex gap-3 text-xs opacity-50">
                       <span className="text-[#666] shrink-0 font-mono text-[10px] mt-0.5 w-[50px]">8m AGO</span>
                       <span className="text-[#ccc]">NVDA supplier TSMC reports minimal disruption from local power fluctuations. Risk downgraded.</span>
                    </div>
                    <div className="flex gap-3 text-xs opacity-40">
                       <span className="text-[#666] shrink-0 font-mono text-[10px] mt-0.5 w-[50px]">12m AGO</span>
                       <span className="text-[#ccc]">Analyzing impact of potential Fed pause on high-duration tech equities...</span>
                    </div>
                    
                    <div className="absolute bottom-0 left-0 w-full h-12 bg-gradient-to-t from-[#111] to-transparent pointer-events-none"></div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Bottom Area: Query Builder */}
          <div className="p-3.5 border-t border-[#1e1e1e] shrink-0 bg-[#0d0d0d] mt-auto">
            <div className="mb-3 border-b border-[#1e1e1e] pb-3">
              <div className="flex items-center justify-between mb-2">
                <p className="text-[10px] text-[#444] uppercase tracking-widest font-medium">query builder</p>
                <span className={`text-[10px] ${isScrapeLive ? 'text-[#1D9E75]' : 'text-[#666]'}`}>{isScrapeLive ? 'Live scrape connected' : 'Waiting for backend...'}</span>
              </div>
              <div className="flex flex-wrap gap-1.5 mb-2.5">
                {['NVDA', 'AAPL', 'GOLD', 'TSLA', 'EURUSD'].map(a => (
                  <span 
                    key={a}
                    onClick={() => toggleAsset(a)}
                    className={`text-[11px] px-3 py-1 rounded-full border cursor-pointer transition-all ${
                      activeAssets.has(a) 
                        ? 'border-[#1D9E75] text-[#1D9E75] bg-[#0a2e1c]' 
                        : 'border-[#2a2a2a] text-[#888] bg-[#161616] hover:border-[#444]'
                    }`}
                  >
                    {a}
                  </span>
                ))}
                <span onClick={addCustomAsset} className="text-[11px] px-3 py-1 rounded-full border border-dashed border-[#2a2a2a] text-[#888] cursor-pointer hover:border-[#EF9F27] hover:text-[#EF9F27]">
                  + add
                </span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {['full report', 'crash risk', 'alpha signals', 'DCF valuation', 'hold window'].map(m => (
                  <span 
                    key={m}
                    onClick={() => toggleMode(m)}
                    className={`text-[11px] px-3 py-1 rounded-full border cursor-pointer transition-all ${
                      activeMode === m 
                        ? 'border-[#378ADD] text-[#378ADD] bg-[#0a1a2e]' 
                        : 'border-[#2a2a2a] text-[#888] bg-[#161616] hover:border-[#444]'
                    }`}
                  >
                    {m}
                  </span>
                ))}
              </div>
            </div>

            <div className="flex flex-wrap gap-1.5">
              <span className="text-[10px] px-2.5 py-1 rounded-full border border-[#1e1e1e] text-[#666] cursor-pointer hover:bg-[#222] hover:text-[#aaa]" onClick={() => fillQ('What is the crash risk for NVDA right now?')}>crash risk NVDA</span>
              <span className="text-[10px] px-2.5 py-1 rounded-full border border-[#1e1e1e] text-[#666] cursor-pointer hover:bg-[#222] hover:text-[#aaa]" onClick={() => fillQ('Show alpha signals for AAPL with historical win rates')}>AAPL alphas</span>
              <span className="text-[10px] px-2.5 py-1 rounded-full border border-[#1e1e1e] text-[#666] cursor-pointer hover:bg-[#222] hover:text-[#aaa]" onClick={() => fillQ('Compare NVDA vs AAPL valuation multiples')}>compare NVDA AAPL</span>
              <span className="text-[10px] px-2.5 py-1 rounded-full border border-[#1e1e1e] text-[#666] cursor-pointer hover:bg-[#222] hover:text-[#aaa]" onClick={() => fillQ('Ask Cortex about ' + (selectedContext && 'label' in selectedContext ? selectedContext.label : 'this asset'))}>Ask Context -&gt;</span>
            </div>
          </div>
        </div>
      </div>

      {/* Floating Chat Box */}
      <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 w-full max-w-2xl px-4 pointer-events-none flex flex-col justify-end">
        {isChatOpen && messages.length > 0 && (
          <div 
            className="bg-[#1a1a1a]/95 backdrop-blur-xl border border-[#333] shadow-[0_8px_32px_rgba(0,0,0,0.8)] rounded-3xl p-4 mb-3 pointer-events-auto max-h-[400px] overflow-y-auto custom-scrollbar flex flex-col gap-3"
            ref={scrollRef}
          >
            {messages.map((msg) => (
              <div key={msg.id} className={`max-w-[90%] ${
                msg.isUser ? 'self-end' : 'self-start'
              }`}>
                {msg.isUser ? (
                  <div className="px-3 py-2.5 rounded-xl text-[13px] leading-relaxed bg-[#222] text-[#eee] border border-[#333]">
                    {msg.content}
                  </div>
                ) : msg.type === 'thinking' ? (
                  <div className="flex items-center gap-1.5 px-3 py-2 bg-[#111] border border-[#1e1e1e] rounded-lg self-start">
                    <div className="w-1.5 h-1.5 rounded-full bg-[#1D9E75] animate-pulse"></div>
                    <div className="w-1.5 h-1.5 rounded-full bg-[#1D9E75] animate-pulse" style={{animationDelay: '0.2s'}}></div>
                    <div className="w-1.5 h-1.5 rounded-full bg-[#1D9E75] animate-pulse" style={{animationDelay: '0.4s'}}></div>
                    <span className="text-[11px] text-[#444] ml-1">22 agents thinking...</span>
                  </div>
                ) : msg.type === 'alert_critical' ? (
                  <div className="bg-[#1a0e0e] border border-[#3d1010] rounded-lg px-3 py-2.5 flex gap-2.5 items-start">
                    <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5 text-[#E24B4A]" />
                    <div className="text-[11px] text-[#ccc] leading-relaxed">
                      <div className="font-medium text-[#E24B4A] mb-1">Critical - exec departure signal</div>
                      NVDA CFO title updated on LinkedIn 3 hours ago. No 8-K filed yet. <br/> Alpha window open - historically stock moves 4-8% on CFO departure news within 24 hours.
                    </div>
                  </div>
                ) : msg.type === 'alert_high' ? (
                  <div className="bg-[#1a1400] border border-[#3d2e00] rounded-lg px-3 py-2.5 flex gap-2.5 items-start">
                    <TrendingUp className="w-3.5 h-3.5 shrink-0 mt-0.5 text-[#EF9F27]" />
                    <div className="text-[11px] text-[#ccc] leading-relaxed">
                      <div className="font-medium text-[#EF9F27] mb-1">High - insider cluster buy</div>
                      3 AAPL executives bought $4.2M open market in 12 days. Historical win rate: 71% over 6 months. Signal strength: 8.5/10.
                    </div>
                  </div>
                ) : msg.type === 'alert_watch' ? (
                  <div className="bg-[#0a1120] border border-[#0d2540] rounded-lg px-3 py-2.5 flex gap-2.5 items-start">
                    <Globe className="w-3.5 h-3.5 shrink-0 mt-0.5 text-[#378ADD]" />
                    <div className="text-[11px] text-[#ccc] leading-relaxed">
                      <div className="font-medium text-[#378ADD] mb-1">Watch - macro shift</div>
                      Fed language analysis: "pause" replaced "hike" in 3 consecutive statements scraped via Bright Data. Historically precedes pivot by 2-3 meetings.
                    </div>
                  </div>
                ) : msg.type === 'crash' ? (
                  <div className="px-3 py-2.5 rounded-xl text-[13px] leading-relaxed bg-[#161616] text-[#ccc] border border-[#2a2a2a]">
                    <div className="text-[11px] font-medium text-white mb-2">
                      CrashPredictor <span className="inline-block text-[9px] font-medium px-1.5 py-0.5 rounded ml-1 bg-[#0a1a2e] text-[#378ADD]">12 indicators</span>
                    </div>
                    <div className="my-2 grid grid-cols-2 gap-1.5 text-[10px]">
                      <span className="text-[#1D9E75]">* VIX: 18.4 - clear</span>
                      <span className="text-[#1D9E75]">* Yield curve: normal</span>
                      <span className="text-[#EF9F27]">* Credit spreads: 380bps</span>
                      <span className="text-[#1D9E75]">* Insider selling: low</span>
                      <span className="text-[#EF9F27]">* PMI: 50.8 - edge</span>
                      <span className="text-[#1D9E75]">* Fed language: neutral</span>
                    </div>
                    <div className="mt-2 p-1.5 bg-[#0a1f10] rounded-md text-[11px] text-[#1D9E75]">
                      2 / 12 indicators RED - CLEAR. Exit trigger: VIX &gt; 32 AND credit spreads &gt; 450bps simultaneously.
                    </div>
                  </div>
                ) : msg.type === 'alpha' ? (
                  <div className="px-3 py-2.5 rounded-xl text-[13px] leading-relaxed bg-[#161616] text-[#ccc] border border-[#2a2a2a]">
                    <div className="text-[11px] font-medium text-white mb-2">
                      AlphaCalculator <span className="inline-block text-[9px] font-medium px-1.5 py-0.5 rounded ml-1 bg-[#0a2e1c] text-[#1D9E75]">3 active</span>
                    </div>
                    <div className="mt-1.5 flex flex-col gap-1.5 text-[11px]">
                      <div className="bg-[#111] border border-[#1e1e1e] rounded-md p-1.5">
                        <div className="text-white font-medium mb-0.5">Insider cluster buy <span className="text-[#1D9E75] float-right">71% win rate</span></div>
                        <div className="text-[#888]">3 execs, $4.2M in 12 days - signal strength 8.5/10</div>
                      </div>
                      <div className="bg-[#111] border border-[#1e1e1e] rounded-md p-1.5">
                        <div className="text-white font-medium mb-0.5">Post-earnings drift <span className="text-[#1D9E75] float-right">63% win rate</span></div>
                        <div className="text-[#888]">EPS beat 5.2% - drift continues 63% of time for 30 days</div>
                      </div>
                      <div className="bg-[#111] border border-[#1e1e1e] rounded-md p-1.5">
                        <div className="text-white font-medium mb-0.5">Smart money convergence <span className="text-[#1D9E75] float-right">74% win rate</span></div>
                        <div className="text-[#888]">Druckenmiller + Tiger Global both added last quarter</div>
                      </div>
                    </div>
                  </div>
                ) : msg.type === 'geo' ? (
                  <div className="px-3 py-2.5 rounded-xl text-[13px] leading-relaxed bg-[#161616] text-[#ccc] border border-[#2a2a2a]">
                    <div className="text-[11px] font-medium text-white mb-2">
                      GeoRiskRadar <span className="inline-block text-[9px] font-medium px-1.5 py-0.5 rounded ml-1 bg-[#2e0a0a] text-[#E24B4A]">elevated</span>
                    </div>
                    <div className="mt-1.5 text-[11px] text-[#aaa] leading-[1.5]">
                      Middle East tensions elevated. Strait of Hormuz shipping data shows 12% slowdown. Historical: 2019 Aramco attack - oil +15% in 1 day, reversed in 2 weeks. Playbook: buy energy short-term, fade after 10 days.
                    </div>
                    <div className="flex gap-2.5 mt-3">
                      <div className="bg-[#111] rounded flex-1 text-center py-1.5 px-2">
                        <div className="text-[8px] text-[#555] mb-0.5 uppercase tracking-widest">Oil impact</div>
                        <div className="text-[11px] font-medium text-[#EF9F27]">+12%</div>
                      </div>
                      <div className="bg-[#111] rounded flex-1 text-center py-1.5 px-2">
                        <div className="text-[8px] text-[#555] mb-0.5 uppercase tracking-widest">Airlines</div>
                        <div className="text-[11px] font-medium text-[#E24B4A]">-8%</div>
                      </div>
                      <div className="bg-[#111] rounded flex-1 text-center py-1.5 px-2">
                        <div className="text-[8px] text-[#555] mb-0.5 uppercase tracking-widest">Gold</div>
                        <div className="text-[11px] font-medium text-[#EF9F27]">+6%</div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="px-3 py-2.5 rounded-xl text-[13px] leading-relaxed bg-[#161616] text-[#ccc] border border-[#2a2a2a]">
                    <div className="text-[11px] font-medium text-white mb-1.5 flex items-center">
                      {msg.assetName || Array.from(activeAssets)[0]} 
                      <span className="inline-block text-[9px] font-medium px-1.5 py-0.5 rounded ml-2 bg-[#0a2e1c] text-[#1D9E75]">BUY</span>
                      <span className="inline-block text-[9px] font-medium px-1.5 py-0.5 rounded ml-1 bg-[#0a2e1c] text-[#1D9E75]">82% conf</span>
                    </div>
                    <div className="text-[11px] text-[#666] mb-2">{msg.modeName || activeMode}</div>
                    <div className="flex gap-2.5 mt-2">
                      <div className="bg-[#111] rounded flex-1 text-center py-1.5 px-2">
                        <div className="text-[8px] text-[#555] mb-0.5 uppercase tracking-widest">DCF PT</div>
                        <div className="text-[11px] font-medium text-[#EF9F27]">$950</div>
                      </div>
                      <div className="bg-[#111] rounded flex-1 text-center py-1.5 px-2">
                        <div className="text-[8px] text-[#555] mb-0.5 uppercase tracking-widest">Upside</div>
                        <div className="text-[11px] font-medium text-[#EF9F27]">+24%</div>
                      </div>
                      <div className="bg-[#111] rounded flex-1 text-center py-1.5 px-2">
                        <div className="text-[8px] text-[#555] mb-0.5 uppercase tracking-widest">Hold</div>
                        <div className="text-[11px] font-medium text-[#EF9F27]">45d</div>
                      </div>
                    </div>
                    <div className="mt-2.5 text-[11px] text-[#aaa]">
                      Exit trigger: gross margin below 68% OR VIX breaks 32 with credit spreads widening. Stop loss: 9% below entry.
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        <div className="bg-[#1a1a1a]/95 backdrop-blur-xl border border-[#333] shadow-[0_8px_32px_rgba(0,0,0,0.8)] rounded-3xl p-1.5 flex items-end pointer-events-auto transition-all focus-within:border-[#C5A059]">
          <textarea 
            rows={1}
            value={inputValue}
            onChange={e => {
              setInputValue(e.target.value);
              e.target.style.height = 'auto';
              e.target.style.height = Math.min(e.target.scrollHeight, 150) + 'px';
            }}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            onClick={() => setIsChatOpen(true)}
            placeholder={placeholder}
            className="flex-1 bg-transparent text-[#eee] text-[15px] px-5 py-3.5 resize-none outline-none font-light placeholder:text-[#666] leading-relaxed custom-scrollbar max-h-[150px]"
          />
          {isChatOpen && (
            <button
               onClick={() => setIsChatOpen(false)}
               className="mb-2 mr-2 text-[#666] hover:text-[#eee] transition-colors p-2 rounded-full hover:bg-[#333]"
               title="Minimize Chat"
            >
               <X className="w-5 h-5" />
            </button>
          )}
          <button 
            onClick={handleSend}
            className={`w-11 h-11 rounded-full flex items-center justify-center shrink-0 mb-1 mr-1 transition-colors ${
              inputValue.trim() ? 'bg-[#C5A059] hover:bg-[#d4b472] text-black' : 'bg-[#2a2a2a] text-[#555]'
            }`}
          >
            <ArrowUp className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
}
