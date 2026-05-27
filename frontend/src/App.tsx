import { useState } from 'react';
import NetworkGraph from './components/NetworkGraph';
import IntelligencePanel from './components/IntelligencePanel';
import NodeDetails from './components/NodeDetails';
import ReportAttachment from './components/ReportAttachment';
import ReportViewer from './components/ReportViewer';
import TradingDashboard from './components/TradingDashboard';
import { GraphNode, Report } from './types';
import { FileText, Copy } from 'lucide-react';

const PREVIOUS_REPORTS: Report[] = [
  {
    id: 'rpt-prev-1',
    title: 'NVDA Alpha Setup',
    subtitle: 'Institutional Review',
    format: 'MD',
    date: 'Oct 23, 14:02 UTC',
    content: '## Post-Earnings Drift Analysis\n\nIdentified an edge based on NVDA post-earnings reactions across 3 prior quarters...\n\n- Historical Win Rate: 72%\n- Average Move: +4.8% \n- Recommended Action: Hold for window of 45 days.\n\nDeepSeek Reasoner has validated fundamental variables are still intact despite geopolitical noise in early Asian trading.'
  },
  {
    id: 'rpt-prev-2',
    title: 'Macro Context: Rate Shock',
    subtitle: 'Crash Predictor Output',
    format: 'MD',
    date: 'Oct 22, 09:12 UTC',
    content: '## Early Warning Flag\n\nVIX and yield curves exhibit structural decoupling typical of late-stage credit contractions. Our 12-factor crash predictor maps this similarly to Q4 2018 behavior.\n\n- Volatility expansion in high beta equities\n- Safe Havens recommended: USD, CHF\n\nMonitor daily options flows closely.'
  }
];

export default function App() {
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [viewingReport, setViewingReport] = useState<Report | null>(null);
  const [leftView, setLeftView] = useState<'network' | 'dashboard'>('network');
  const [isDashFullscreen, setIsDashFullscreen] = useState(false);

  const selectedContext = selectedNode || viewingReport;
  
  let currentSymbol = 'NVDA';
  if (selectedNode && selectedNode.group === 'Ticker') {
    currentSymbol = selectedNode.label;
  } else if (viewingReport) {
    currentSymbol = viewingReport.title.split(' ')[0] || 'NVDA';
  }

  const handleClearContext = () => {
    setSelectedNode(null);
    setViewingReport(null);
  };

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-[#121212] text-[#E0DCD9] font-sans selection:bg-[#C5A059]/30 selection:text-white relative">
      {/* Top Navigation */}
      <header className="h-14 border-b border-[#333] flex items-center justify-between px-6 shrink-0 bg-[#1a1a1a] relative z-10">
        <div className="flex items-center gap-4">
          <div className="font-light text-sm tracking-[0.3em] uppercase text-white">
            Cortex
          </div>
          <div className="ml-8 flex bg-[#121212] rounded-lg p-1 border border-[#333]">
            <button 
              onClick={() => setLeftView('network')}
              className={`px-4 py-1.5 rounded-md text-[10px] font-bold uppercase tracking-widest transition-colors ${leftView === 'network' ? 'bg-[#2a2a2a] text-white' : 'text-[#666] hover:text-[#ccc]'}`}
            >
              Neural DB
            </button>
            <button 
              onClick={() => setLeftView('dashboard')}
              className={`px-4 py-1.5 rounded-md text-[10px] font-bold uppercase tracking-widest transition-colors ${leftView === 'dashboard' ? 'bg-[#2a2a2a] text-white' : 'text-[#666] hover:text-[#ccc]'}`}
            >
              Live Monitor
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 flex overflow-hidden p-6 gap-6 pb-24">
        
        {/* Left Column: Graph & Reports List */}
        <div className={`flex flex-col min-w-0 gap-6 transition-all duration-500 ease-in-out ${leftView === 'dashboard' ? 'flex-[60%]' : 'flex-[50%]'}`}>
          <div className={`relative flex flex-col min-h-0 ${leftView === 'dashboard' ? 'flex-1' : 'flex-[60%]'}`}>
            {leftView === 'network' ? (
              <>
                <NetworkGraph onNodeClick={setSelectedNode} />
                {selectedNode && (
                  <NodeDetails node={selectedNode} onClose={() => setSelectedNode(null)} />
                )}
              </>
            ) : (
              <>
                <TradingDashboard 
                  symbol={currentSymbol} 
                  isFullscreen={isDashFullscreen}
                  onToggleFullscreen={() => setIsDashFullscreen(prev => !prev)}
                />
                {/* Fullscreen Backdrop Overlay */}
                {isDashFullscreen && leftView === 'dashboard' && (
                   <div className="fixed inset-0 z-[90] bg-black/80 backdrop-blur-md"></div>
                )}
              </>
            )}
          </div>
          
          {leftView === 'network' && (
          <div className="flex-[40%] flex flex-col min-h-0 bg-[#1a1a1a] rounded-xl border border-[#333] p-4">
            <h3 className="text-[10px] uppercase tracking-[0.2em] text-[#666] mb-4 pl-1 font-medium flex items-center gap-2 max-w-fit">
              <FileText className="w-3 h-3 text-[#C5A059]" />
              Generated Reports & Documents
            </h3>
            <div className="flex-1 overflow-y-auto space-y-3 custom-scrollbar pr-2 pb-2">
               {PREVIOUS_REPORTS.map(rpt => (
                  <div key={rpt.id}>
                    <ReportAttachment report={rpt} onClick={setViewingReport} />
                  </div>
               ))}
               <div className="bg-[#111] border border-[#222] rounded-xl p-3 flex flex-col">
                 <div className="flex items-center justify-between mb-2">
                   <span className="text-[#eee] text-xs font-semibold">SEC Form 10-K Document Scan</span>
                   <Copy className="w-3 h-3 text-[#666] hover:text-[#eee] cursor-pointer" onClick={() => navigator.clipboard.writeText('SEC Form 10-K Document Scan')} />
                 </div>
                 <span className="text-[#888] text-[11px]">System digested full document in 1.2s. Extracted risk factors matching current regime...</span>
               </div>
            </div>
          </div>
          )}
        </div>

        {/* Right Column: Complete Chat Interface */}
        <div className={`flex flex-col min-w-0 h-full transition-all duration-500 ease-in-out ${leftView === 'dashboard' ? 'flex-[40%]' : 'flex-[50%]'}`}>
          <IntelligencePanel 
            selectedContext={selectedContext}
            onClearContext={handleClearContext}
            isPanelVisible={true}
            viewMode={leftView}
          />
        </div>
      </main>
    </div>
  );
}
