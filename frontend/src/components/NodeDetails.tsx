import { GraphNode } from '../types';
import { X } from 'lucide-react';

interface NodeDetailsProps {
  node: GraphNode;
  onClose: () => void;
}

export default function NodeDetails({ node, onClose }: NodeDetailsProps) {
  return (
    <div className="flex-1 w-full bg-[#111] rounded-2xl border border-[#1e1e1e] flex flex-col overflow-hidden font-sans">
      <div className="flex justify-between items-center p-6 border-b border-[#1e1e1e]">
        <h3 className="text-[10px] uppercase tracking-[0.2em] text-[#888]">Node Details</h3>
        <div className="flex items-center gap-4">
          <span className="text-[#C5A059] text-[9px] uppercase tracking-[0.2em] font-medium border border-[#C5A059]/30 bg-[#C5A059]/10 px-2 py-0.5 rounded-sm">
            {node.group}
          </span>
          <button onClick={onClose} className="p-1 text-[#666] hover:text-[#eee] transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
      
      <div className="p-6 space-y-6 text-sm">
        <div className="grid grid-cols-[80px_1fr] gap-3 items-baseline">
          <span className="text-[#666] text-[10px] uppercase tracking-widest">Name</span>
          <span className="text-[#eee] font-medium text-sm">{node.label}</span>
          
          <span className="text-[#666] text-[10px] uppercase tracking-widest">Type</span>
          <span className="text-[#888] font-mono text-[10px] break-all">{node.group === 'Ticker' ? 'Asset' : 'Agent Processor'}</span>
          
          {node.conviction !== undefined && (
            <>
              <span className="text-[#666] text-[10px] uppercase tracking-widest">Conviction</span>
              <span className="text-[#888] text-xs font-light">{(node.conviction * 100).toFixed(0)}%</span>
            </>
          )}

          {node.signalDirection && (
            <>
              <span className="text-[#666] text-[10px] uppercase tracking-widest">Signal</span>
              <span className={`text-xs font-bold uppercase tracking-widest ${
                node.signalDirection === 'bullish' ? 'text-[#059669]' : 
                node.signalDirection === 'bearish' ? 'text-[#ef4444]' : 'text-[#fb923c]'
              }`}>
                {node.signalDirection}
              </span>
            </>
          )}
        </div>

        {node.group !== 'Ticker' && (
          <div className="pt-6 border-t border-[#333] flex flex-col items-center">
            <div className="w-12 h-12 rounded-full border border-[#C5A059] flex items-center justify-center relative mb-4">
              <div className="w-4 h-4 bg-[#C5A059] rounded-full animate-pulse"></div>
              <div className="absolute inset-0 rounded-full border border-[#C5A059] opacity-50 animate-ping"></div>
            </div>
            <div className="text-[10px] uppercase tracking-widest text-[#888] text-center">
              Awaiting next simulation tick...
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
