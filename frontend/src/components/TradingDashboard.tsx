import { useId } from 'react';
import { Maximize2, Minimize2 } from 'lucide-react';

const equityData = [
  { time: 'May 13', val: 994 },
  { time: 'May 14', val: 998 },
  { time: 'May 15', val: 1001 },
  { time: 'May 16', val: 1003 },
  { time: 'May 17', val: 1005 },
  { time: 'May 18', val: 1008 },
  { time: 'May 19', val: 1010 }
];

const baselineData = [
  { time: '1', val: 100 }, { time: '2', val: 105 }, { time: '3', val: 102 }, 
  { time: '4', val: 108 }, { time: '5', val: 112 }, { time: '6', val: 110 }, { time: '7', val: 115 }
];

const recentActivity = [
  { time: '19 May 12:15:22', symbol: 'LTC/USD', side: 'BUY', type: 'DIAGNOSTIC', status: 'FILLED', pnl: '+0.18', pnlColor: 'text-[#1D9E75]' },
  { time: '19 May 12:13:10', symbol: 'ETH/USD', side: 'BUY', type: 'DIAGNOSTIC', status: 'FILLED', pnl: '-0.12', pnlColor: 'text-[#E24B4A]' },
  { time: '19 May 12:10:01', symbol: 'SOL/USD', side: 'SELL', type: 'DIAGNOSTIC', status: 'FILLED', pnl: '-0.21', pnlColor: 'text-[#E24B4A]' },
  { time: '19 May 12:07:55', symbol: 'AVAX/USD', side: 'BUY', type: 'DIAGNOSTIC', status: 'FILLED', pnl: '+0.09', pnlColor: 'text-[#1D9E75]' },
  { time: '19 May 12:05:44', symbol: 'BTC/USD', side: 'BUY', type: 'DIAGNOSTIC', status: 'FILLED', pnl: '-0.49', pnlColor: 'text-[#E24B4A]' }
];

interface TradingDashboardProps {
  symbol?: string;
  isFullscreen?: boolean;
  onToggleFullscreen?: () => void;
}

function MiniLineChart({
  data,
  stroke = '#1D9E75',
  showAxes = false,
  showArea = false,
}: {
  data: Array<{ time: string; val: number }>;
  stroke?: string;
  showAxes?: boolean;
  showArea?: boolean;
}) {
  const gradientId = useId().replace(/:/g, '-');
  const width = 100;
  const height = 100;
  const padding = showAxes ? 10 : 4;
  const innerWidth = width - padding * 2;
  const innerHeight = height - padding * 2;
  const values = data.map((point) => point.val);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const stepX = data.length > 1 ? innerWidth / (data.length - 1) : 0;

  const points = data.map((point, index) => {
    const x = padding + index * stepX;
    const y = padding + innerHeight - ((point.val - min) / range) * innerHeight;
    return { x, y };
  });

  const linePath = points
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`)
    .join(' ');

  const areaPath = `${linePath} L ${points[points.length - 1].x} ${height - padding} L ${points[0].x} ${height - padding} Z`;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-full w-full" aria-hidden="true">
      <defs>
        <linearGradient id={gradientId} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={stroke} stopOpacity={showArea ? 0.3 : 0} />
          <stop offset="100%" stopColor={stroke} stopOpacity={0} />
        </linearGradient>
      </defs>
      {showAxes && (
        <>
          <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#1e232e" strokeWidth="0.8" />
          <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#1e232e" strokeWidth="0.8" />
        </>
      )}
      {showArea && <path d={areaPath} fill={`url(#${gradientId})`} />}
      <path d={linePath} fill="none" stroke={stroke} strokeWidth="2.2" strokeLinejoin="round" strokeLinecap="round" />
      {!showAxes &&
        points.map((point, index) => (
          <circle key={index} cx={point.x} cy={point.y} r="1.9" fill={stroke} />
        ))}
    </svg>
  );
}

export default function TradingDashboard({ symbol = 'NVDA', isFullscreen, onToggleFullscreen }: TradingDashboardProps) {
  const urlSymbol = encodeURIComponent(symbol);

  return (
    <div className={`w-full flex-col bg-[#0A0D14] rounded-xl border border-[#1e232e] overflow-y-auto custom-scrollbar p-5 font-sans leading-relaxed shadow-lg ${isFullscreen ? 'fixed inset-4 z-[100] flex' : 'flex-1 flex'}`}>
      <div className="flex items-center justify-between mb-4 shrink-0">
        <h2 className="text-[#1D9E75] font-semibold tracking-wider text-xs uppercase">Live Monitor Analysis</h2>
        <div className="flex items-center gap-3">
          {onToggleFullscreen && (
            <button onClick={onToggleFullscreen} className="p-1.5 hover:bg-[#121620] border border-transparent hover:border-[#1e232e] rounded transition-colors text-[#ccc]">
              {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
            </button>
          )}
        </div>
      </div>

      <div className="bg-[#121620] border border-[#1e232e] rounded-xl mb-4 shrink-0 h-[450px] overflow-hidden flex flex-col">
        <div className="px-5 py-4 border-b border-[#1e232e] flex justify-between items-center bg-[#0A0D14]/50">
          <h3 className="text-[#eee] text-sm font-medium uppercase tracking-widest">{symbol} Watch</h3>
          <span className="text-[10px] text-[#888] tracking-widest uppercase">TradingView Live</span>
        </div>
        <div className="flex-1 w-full relative pt-2">
          <iframe
            src={`https://s.tradingview.com/widgetembed/?symbol=${urlSymbol}&interval=D&hidesidetoolbar=0&symboledit=1&saveimage=1&toolbarbg=121620&studies=%5B%5D&theme=dark&style=1&timezone=Etc%2FUTC&studies_overrides=%7B%7D&overrides=%7B%7D&enabled_features=%5B%5D&disabled_features=%5B%5D&locale=en&utm_source=tradingview.com&utm_medium=widget_new&utm_campaign=chart&utm_term=${urlSymbol}`}
            className="absolute inset-0 w-full h-full"
            frameBorder="0"
            allowtransparency="true"
            scrolling="no"
            allowFullScreen={true}
          ></iframe>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-4 shrink-0 h-64">
        <div className="col-span-2 bg-[#121620] border border-[#1e232e] rounded-xl p-5 flex flex-col h-full">
          <h3 className="text-[10px] text-[#888] uppercase tracking-wider mb-4">Equity Curve</h3>
          <div className="flex-1 min-h-0 w-full">
            <MiniLineChart data={equityData} showAxes showArea />
            <div className="mt-3 flex justify-between text-[10px] uppercase tracking-wider text-[#555]">
              {equityData.map((point) => (
                <span key={point.time}>{point.time}</span>
              ))}
            </div>
          </div>
        </div>
        <div className="col-span-1 bg-[#121620] border border-[#1e232e] rounded-xl p-5 flex flex-col h-full">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-[10px] text-[#888] uppercase tracking-wider">Top Positions</h3>
            <span className="text-[10px] text-[#888] uppercase tracking-wider">PNL (USD)</span>
          </div>
          <div className="flex-1 overflow-y-auto space-y-4">
            {[
              {s: 'AAPLx', p: '+7.21'},
              {s: 'TSLAx', p: '+4.88'},
              {s: 'NVDAx', p: '+3.32'},
              {s: 'AMZNx', p: '+2.11'},
              {s: 'MSFTx', p: '+1.66'}
            ].map(item => (
              <div key={item.s} className="flex justify-between items-center">
                <span className="text-[#ccc] text-sm font-medium">{item.s}</span>
                <span className="text-[#1D9E75] text-sm">{item.p}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4 shrink-0">
        <div className="bg-[#121620] border border-[#1e232e] rounded-xl p-5">
           <h3 className="text-[10px] text-[#888] uppercase tracking-wider mb-4">30-Day Baseline</h3>
           <div className="flex items-end justify-between border-b border-[#1e232e] pb-3 mb-3">
              <div className="flex flex-col">
                <span className="text-[10px] text-[#888] uppercase tracking-wider mb-1">PNL (USD)</span>
                <span className="text-2xl font-semibold text-[#1D9E75]">+33.56</span>
              </div>
              <div className="w-24 h-8 mb-2">
                <MiniLineChart data={baselineData} />
              </div>
           </div>
           <div className="flex justify-between">
              <div className="flex flex-col">
                <span className="text-[10px] text-[#888] uppercase tracking-wider mb-1">Trades</span>
                <span className="text-xl font-medium text-white tracking-tight">138</span>
              </div>
              <div className="flex flex-col text-center">
                <span className="text-[10px] text-[#888] uppercase tracking-wider mb-1">Win Rate</span>
                <span className="text-xl font-medium text-white tracking-tight">53%</span>
              </div>
              <div className="flex flex-col text-right">
                <span className="text-[10px] text-[#888] uppercase tracking-wider mb-1">Max Drawdown</span>
                <span className="text-xl font-medium text-white tracking-tight">1.68%</span>
              </div>
           </div>
        </div>

        <div className="bg-[#121620] border border-[#1e232e] rounded-xl p-5 flex flex-col justify-between">
           <h3 className="text-[10px] text-[#888] uppercase tracking-wider">Walk-Forward (OOS)</h3>
           <div className="flex flex-col mt-4">
              <span className="text-[10px] text-[#888] uppercase tracking-wider mb-1">Surviving Configs</span>
              <span className="text-3xl font-semibold text-[#E24B4A]">0</span>
           </div>
           <div className="mt-4 flex items-center justify-between">
             <span className="text-xs text-[#888] leading-tight">Reported honestly.<br/>No curve-fitting.</span>
             <div className="w-32 h-2 flex gap-[2px]">
               {Array.from({length: 15}).map((_, i) => (
                 <div key={i} className="flex-1 bg-[#E24B4A] opacity-60 rounded-full h-[2px]"></div>
               ))}
             </div>
           </div>
        </div>
      </div>

      <div className="bg-[#121620] border border-[#1e232e] rounded-xl p-5 shrink-0 mb-4">
        <h3 className="text-[10px] text-[#888] uppercase tracking-wider mb-4">Recent Activity (Dry Run / Diagnostic)</h3>
        <div className="w-full">
          <div className="grid grid-cols-6 gap-4 text-[10px] text-[#888] uppercase tracking-wider mb-3">
             <span className="col-span-2">Time (UTC)</span>
             <span>Symbol</span>
             <span>Side</span>
             <span>Type</span>
             <span>Status</span>
             <span className="text-right">PNL (USD)</span>
          </div>
          <div className="space-y-3">
            {recentActivity.map((act, i) => (
              <div key={i} className="grid grid-cols-6 gap-4 text-xs">
                <span className="col-span-2 text-[#aaa]">{act.time}</span>
                <span className="text-[#eee]">{act.symbol}</span>
                <span className={act.side === 'BUY' ? 'text-[#1D9E75]' : 'text-[#E24B4A]'}>{act.side}</span>
                <span className="text-[#888]">{act.type}</span>
                <span className="text-[#1D9E75]">{act.status}</span>
                <span className={`text-right ${act.pnlColor}`}>{act.pnl}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
