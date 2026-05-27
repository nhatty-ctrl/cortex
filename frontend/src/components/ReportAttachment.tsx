import { Report } from '../types';
import { FileText } from 'lucide-react';

interface ReportAttachmentProps {
  report: Report;
  onClick: (report: Report) => void;
}

export default function ReportAttachment({ report, onClick }: ReportAttachmentProps) {
  // Simple extraction of first coherent sentence for AI summary simulation
  const getSimulatedSummary = (content: string) => {
    const textObj = content.replace(/[#*\-]/g, '').trim().split('\n').filter(l => l.length > 20);
    const summary = textObj.length > 0 ? textObj[0].slice(0, 50) + '...' : 'Analysis ready...';
    return `AI Summary: ${summary}`;
  };

  return (
    <div 
      onClick={() => onClick(report)}
      className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl p-3 flex items-center justify-between cursor-pointer hover:bg-[#222] transition-colors group shadow-sm"
    >
      <div className="flex items-center gap-3">
        <div className="w-8 h-10 bg-[#222] rounded flex items-center justify-center border border-[#333] shadow-inner shrink-0 relative overflow-hidden group-hover:border-[#444] transition-colors">
          <FileText className="w-4 h-4 text-[#888]" />
        </div>
        <div className="flex flex-col overflow-hidden">
          <span className="text-[#ddd] font-medium font-sans text-[13px] group-hover:text-white transition-colors leading-tight truncate">{report.title}</span>
          <span className="text-[#777] text-[10px] mt-0.5 font-medium tracking-wide uppercase truncate">{report.subtitle}</span>
          <span className="text-[#C5A059] text-[11px] mt-1.5 font-light truncate opacity-80">{getSimulatedSummary(report.content)}</span>
        </div>
      </div>
    </div>
  );
}
