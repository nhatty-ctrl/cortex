import { Report } from '../types';
import { X, Copy, HardDrive } from 'lucide-react';

interface ReportViewerProps {
  report: Report;
  onClose: () => void;
}

export default function ReportViewer({ report, onClose }: ReportViewerProps) {
  const isNVDA = report.id === 'rpt-prev-1';

  return (
    <div className="w-full bg-[#111] rounded-2xl border border-[#1e1e1e] flex flex-col font-sans relative shrink-0">
        
        {/* Top Navigation Bar */}
        <div className="h-16 border-b border-[#222] px-6 flex items-center justify-between shrink-0">
          <div className="flex flex-col">
            <div className="flex items-center gap-3">
              <span className="text-xl text-[#eee] font-semibold tracking-tight">{report.title}</span>
            </div>
            <span className="text-[10px] text-[#666] tracking-widest uppercase mt-0.5">{report.subtitle}</span>
          </div>
          
          <div className="flex items-center gap-4">
             {isNVDA && (
               <div className="flex items-center gap-2 mr-4">
                  <span className="bg-[#059669] text-white text-xs font-bold px-3 py-1 rounded">BUY</span>
                  <span className="text-[#059669] font-medium text-sm">PT $950 <span className="text-[#666] text-xs">+24%</span></span>
               </div>
             )}
             {!isNVDA && (
                <button className="flex items-center gap-2 px-4 py-2 border border-[#444] bg-[#222] rounded-lg text-xs font-medium text-[#ccc] hover:bg-[#333] hover:text-white transition-colors">
                  <Copy className="w-3.5 h-3.5" />
                  Copy
                </button>
             )}
             <button onClick={onClose} className="p-2 text-[#666] hover:text-[#eee] transition-colors rounded-full hover:bg-[#333]">
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 p-10 text-[#eee] bg-[#121212] rounded-b-2xl">
          {isNVDA ? (
            <div className="grid grid-cols-2 gap-x-12 gap-y-10 max-w-4xl mx-auto">
              {/* Sec 1 */}
              <div className="space-y-3">
                <h3 className="text-sm font-bold flex items-center gap-2"><span className="text-[#666] font-mono text-xs">1.</span> what happened</h3>
                <p className="text-[13px] text-[#aaa] leading-relaxed">
                  Post-earnings beat: EPS $5.16 vs $4.64 est. Data center revenue +427% YoY. Blackwell chip demand outpacing supply.
                </p>
              </div>
              
              {/* Sec 2 */}
              <div className="space-y-3">
                <h3 className="text-sm font-bold flex items-center gap-2"><span className="text-[#666] font-mono text-xs">2.</span> event chain</h3>
                <p className="text-[13px] text-[#aaa] leading-relaxed">
                  NVDA beat -&gt; AI capex cycle confirmed -&gt; AMD/TSM benefit -&gt; energy grid demand -&gt; copper rally
                </p>
              </div>

              {/* Sec 3 */}
              <div className="space-y-3">
                <h3 className="text-sm font-bold flex items-center gap-2"><span className="text-[#666] font-mono text-xs">3.</span> alpha signals</h3>
                <div className="space-y-2 text-[13px] text-[#aaa]">
                  <div className="flex justify-between border-b border-[#222] pb-1">
                    <span>Insider cluster buy</span>
                    <span className="font-semibold text-[#eee]">71% win rate</span>
                  </div>
                  <div className="flex justify-between border-b border-[#222] pb-1">
                    <span>Post-earnings drift</span>
                    <span className="font-semibold text-[#eee]">63% win rate</span>
                  </div>
                  <div className="flex justify-between border-b border-[#222] pb-1">
                    <span>Smart money convergence</span>
                    <span className="font-semibold text-[#eee]">74% win rate</span>
                  </div>
                </div>
              </div>

              {/* Sec 4 */}
              <div className="space-y-3">
                <h3 className="text-sm font-bold flex items-center gap-2"><span className="text-[#666] font-mono text-xs">4.</span> calculations</h3>
                <div className="space-y-2 text-[13px] text-[#aaa]">
                  <div className="flex justify-between border-b border-[#222] pb-1">
                    <span>ROIC</span>
                    <span className="font-semibold text-[#eee]">42.1%</span>
                  </div>
                  <div className="flex justify-between border-b border-[#222] pb-1">
                    <span>Gross margin</span>
                    <span className="font-semibold text-[#eee]">74.6%</span>
                  </div>
                  <div className="flex justify-between border-b border-[#222] pb-1">
                    <span>DCF intrinsic</span>
                    <span className="font-semibold text-[#eee]">$912/sh</span>
                  </div>
                  <div className="flex justify-between border-b border-[#222] pb-1">
                    <span>VaR 95% 30d</span>
                    <span className="font-semibold text-[#eee]">$1,840</span>
                  </div>
                </div>
              </div>

              {/* Sec 5 */}
              <div className="space-y-3">
                <h3 className="text-sm font-bold flex items-center gap-2"><span className="text-[#666] font-mono text-xs">5.</span> bull / base / bear</h3>
                <div className="flex h-2 rounded-full overflow-hidden mt-3">
                  <div className="bg-[#059669]" style={{ width: '40%' }}></div>
                  <div className="bg-[#fb923c]" style={{ width: '45%' }}></div>
                  <div className="bg-[#ef4444]" style={{ width: '15%' }}></div>
                </div>
                <div className="flex justify-between text-[11px] font-semibold mt-1">
                  <span className="text-[#059669]">Bull 40%</span>
                  <span className="text-[#fb923c]">Base 45%</span>
                  <span className="text-[#ef4444]">Bear 15%</span>
                </div>
              </div>

              {/* Sec 6 */}
              <div className="space-y-3">
                <h3 className="text-sm font-bold flex items-center gap-2"><span className="text-[#666] font-mono text-xs">6.</span> hold window</h3>
                <div className="space-y-2 text-[13px] text-[#aaa]">
                  <div className="flex justify-between border-b border-[#222] pb-1">
                    <span>Hold until</span>
                    <span className="font-semibold text-[#eee]">Q3 earnings Aug 15</span>
                  </div>
                  <div className="flex justify-between border-b border-[#222] pb-1">
                    <span>Stop loss</span>
                    <span className="font-semibold text-[#eee]">$695 (-9%)</span>
                  </div>
                  <div className="flex justify-between border-b border-[#222] pb-1">
                    <span>Exit if</span>
                    <span className="font-semibold text-[#eee]">GM below 68%</span>
                  </div>
                  <div className="flex justify-between border-b border-[#222] pb-1">
                    <span>Pre-crash exit</span>
                    <span className="font-semibold text-[#eee]">VIX &gt; 32 + spreads</span>
                  </div>
                </div>
              </div>

              {/* Sec 7 */}
              <div className="space-y-3">
                <h3 className="text-sm font-bold flex items-center gap-2"><span className="text-[#666] font-mono text-xs">7.</span> historical parallel</h3>
                <p className="text-[13px] text-[#aaa] leading-relaxed">
                  <span className="text-[#eee] font-medium">Matches Jan 2023 AI breakout (74% similarity).</span> What worked: buy on any 10% dip, hold 6 months. Avg return: +38%.
                </p>
              </div>

              {/* Sec 8 */}
              <div className="space-y-3">
                <h3 className="text-sm font-bold flex items-center gap-2"><span className="text-[#666] font-mono text-xs">8.</span> confidence audit</h3>
                <div className="space-y-2 text-[13px] text-[#aaa]">
                  <div className="flex justify-between border-b border-[#222] pb-1">
                    <span>EPS beat</span>
                    <span className="font-semibold text-[#059669] flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-[#059669]"></span> verified</span>
                  </div>
                  <div className="flex justify-between border-b border-[#222] pb-1">
                    <span>ROIC calc</span>
                    <span className="font-semibold text-[#059669] flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-[#059669]"></span> verified</span>
                  </div>
                  <div className="flex justify-between border-b border-[#222] pb-1">
                    <span>DCF value</span>
                    <span className="font-semibold text-[#fb923c] flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-[#fb923c]"></span> 1 source</span>
                  </div>
                  <div className="text-[11px] text-[#555] mt-2 flex items-center gap-1.5 font-mono">
                    <HardDrive className="w-3 h-3" /> Data scraped 4 min ago via Bright Data
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto space-y-6 pt-4 text-[14px] leading-relaxed text-[#ccc]">
              {report.content.split('\n\n').map((paragraph, idx) => {
                if (paragraph.startsWith('##')) {
                  return <h3 key={idx} className="text-xl font-semibold text-white mb-4 mt-8">{paragraph.replace('##', '').trim()}</h3>;
                }
                if (paragraph.startsWith('-')) {
                  return (
                    <ul key={idx} className="list-disc pl-6 space-y-2 text-[#aaa]">
                      {paragraph.split('\n').map((item, i) => (
                         <li key={i}>{item.replace('-', '').trim()}</li>
                      ))}
                    </ul>
                  )
                }
                return <p key={idx}>{paragraph}</p>;
              })}
            </div>
          )}
        </div>
    </div>
  );
}
