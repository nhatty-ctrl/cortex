import { useEffect, useRef, useMemo, useState, type MouseEvent } from 'react';
import * as d3 from 'd3';
import { GraphNode } from '../types';
import { createEventSource, fetchGraph } from '../api';

interface NetworkGraphProps {
  onNodeClick: (node: GraphNode) => void;
}

export default function NetworkGraph({ onNodeClick }: NetworkGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const gRef = useRef<SVGGElement>(null);
  
  const [remoteGraph, setRemoteGraph] = useState<{ nodes: any[]; edges: any[] } | null>(null);
  const mousePos = useRef({ x: 0, y: 0 });
  const activeNodeIdRef = useRef<number | null>(null);

  const { nodes, edges } = useMemo(() => {
    const agents = [
      'Supervisor', 'DeepSeek', 'Filings', 'Social', 'SmartMoney', 'Fundament',
      'DCF', 'Sentiment', 'GeoRisk', 'Risk', 'ExecWatch', 'Playbook', 'Hold',
      'Report', 'Alerts', 'Verify', 'Alpha', 'Scraper', 'Crash', 'Volumes',
      'MacroTracker', 'NewsSweep'
    ];
    const NUM_NODES = agents.length;
    
    const n = [];
    for (let i = 0; i < NUM_NODES; i++) {
      let x = 0;
      let y = 0;
      let isHub = false;
      
      if (i === 0) { // Supervisor
        x = 0;
        y = 0;
        isHub = true;
      } else if (i === 1) { // DeepSeek
        x = -40;
        y = 50;
        isHub = true;
      } else {
        // Arrange in a circle/ellipse
        const angle = (i - 2) * (Math.PI * 2) / (NUM_NODES - 2);
        const radiusX = 220 + Math.random() * 80;
        const radiusY = 180 + Math.random() * 60;
        x = Math.cos(angle) * radiusX;
        y = Math.sin(angle) * radiusY;
      }

      const z = -0.5 + Math.random() * 1;
      const baseAngle = Math.atan2(y, x);
      const dist = Math.sqrt(x*x + y*y);

      let nodeColor = '#888';
      let opacity = 0.5;
      
      if (isHub) {
        nodeColor = '#1D9E75'; // Hub is green
        opacity = 0.9;
      } else {
         const rand = Math.random();
         if (rand > 0.85) {
           nodeColor = '#EF9F27'; // Orange
           opacity = 0.8;
         } else if (rand > 0.70) {
           nodeColor = '#C5A059'; // Gold
           opacity = 0.7;
         } else if (rand > 0.60) {
           nodeColor = '#378ADD'; // Blue
           opacity = 0.8;
         }
      }

      n.push({
        id: i,
        baseAngle,
        dist,
        x, y, z,
        size: isHub ? 4 : 2.5,
        color: nodeColor,
        opacity: Math.max(opacity, 0.4),
        nodeData: {
          id: `agent-${i}`,
          label: agents[i],
          group: 'Analysis',
          conviction: Math.random()
        } as GraphNode
      });
    }
    
    // Connect nodes primarily nearest neighbors for visual web
    const e = [];
    n.forEach((target, i) => {
      if (i > 1) {
        // Connect to Supervisor (0)
        e.push({ source: n[0], target: target });
        
        // 40% chance to also connect to DeepSeek (1)
        if (Math.random() > 0.6) {
           e.push({ source: n[1], target: target });
        }
        
        // Connect to neighbors in ring
        if (Math.random() > 0.7) {
          const nextIdx = i === NUM_NODES - 1 ? 2 : i + 1;
          e.push({ source: n[i], target: n[nextIdx] });
        }

        // Random cross connections
        if (Math.random() > 0.85) {
          const randomIdx = Math.floor(Math.random() * (NUM_NODES - 2)) + 2;
          if (randomIdx !== i) {
             e.push({ source: n[i], target: n[randomIdx] });
          }
        }
      }
    });

    // Add a single edge between Supervisor and DeepSeek
    e.push({ source: n[0], target: n[1] });
    return { nodes: n, edges: e };
  }, []);

  const graphData = useMemo(() => {
    if (!remoteGraph) return { nodes, edges };

    const mappedNodes = remoteGraph.nodes.map((node, index) => {
      const angle = (index / Math.max(1, remoteGraph.nodes.length)) * Math.PI * 2;
      const x = typeof node.x === 'number' ? node.x : Math.cos(angle) * 220;
      const y = typeof node.y === 'number' ? node.y : Math.sin(angle) * 180;
      const z = typeof node.z === 'number' ? node.z : -0.5 + (index / Math.max(1, remoteGraph.nodes.length)) * 1;
      const color = node.color || (node.signal === 'bullish' ? '#1D9E75' : node.signal === 'bearish' ? '#E24B4A' : '#888');
      return {
        id: index,
        baseAngle: Math.atan2(y, x),
        dist: Math.sqrt(x * x + y * y),
        x,
        y,
        z,
        size: node.size ? Math.max(2.5, node.size / 12) : 2.5,
        color,
        opacity: 0.75,
        nodeData: {
          id: node.id,
          label: node.label,
          group: 'Analysis',
          conviction: node.confidence ?? 0.5,
        } as GraphNode,
      };
    });

    const mappedEdges = remoteGraph.edges.map((edge: any) => {
      const sourceIndex = mappedNodes.findIndex(n => n.nodeData.id === edge.source);
      const targetIndex = mappedNodes.findIndex(n => n.nodeData.id === edge.target);
      return {
        source: mappedNodes[Math.max(0, sourceIndex)].id,
        target: mappedNodes[Math.max(0, targetIndex)].id,
        impactMagnitude: edge.impactMagnitude ?? 0.2,
      };
    });

    return { nodes: mappedNodes, edges: mappedEdges };
  }, [remoteGraph, nodes, edges]);

  useEffect(() => {
    if (!svgRef.current || !gRef.current) return;
    const nodes = graphData.nodes;
    const edges = graphData.edges;
    
    // Zoom behavior
    const svg = d3.select(svgRef.current);
    const g = d3.select(gRef.current);
    
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 5])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });
    
    svg.call(zoom);
    // Initial center scale
    svg.call(zoom.transform, d3.zoomIdentity.translate(0, 0).scale(0.8));

    const loadGraph = async () => {
      try {
        const data = await fetchGraph();
        if (data?.nodes && data?.edges) {
          setRemoteGraph({ nodes: data.nodes, edges: data.edges });
        }
      } catch (error) {
        console.warn('Unable to load backend graph', error);
      }
    };

    const source = createEventSource('/api/signals/stream', (event) => {
      if (event?.type === 'graph_update' && event?.nodes && event?.edges) {
        setRemoteGraph({ nodes: event.nodes, edges: event.edges });
      }
    });

    loadGraph();

    // Animation Loop
    let animationFrameId: number;
    let time = 0;

    const render = () => {
      time += 0.0005; // Slower speed
      const mX = mousePos.current.x;
      const mY = mousePos.current.y;
      const activeId = activeNodeIdRef.current;

      const updatedNodes = nodes.map(n => {
        // Subtle revolve based on time + mouse influence
        const currentAngle = n.baseAngle + time * (n.z > 0 ? 1 : -1) * 0.5;
        // The structure revolves slightly, but keeps its original shape 
        // by only blending a small portion of the circular movement
        
        const revolveX = Math.cos(currentAngle) * n.dist;
        const revolveY = Math.sin(currentAngle) * n.dist;

        // Blend original position with revolved position for a "wobble"
        const blend = 0.2; // 20% revolving
        const bx = n.x * (1 - blend) + revolveX * blend;
        const by = n.y * (1 - blend) + revolveY * blend;

        // Parallax depth based on mouse
        const px = bx + mX * 60 * n.z;
        const py = by + mY * 60 * n.z;

        return { ...n, currX: px, currY: py };
      });

      // Update edges
      g.selectAll('line')
        .data(edges)
        .join('line')
        .attr('x1', (d: any) => updatedNodes[d.source.id].currX)
        .attr('y1', (d: any) => updatedNodes[d.source.id].currY)
        .attr('x2', (d: any) => updatedNodes[d.target.id].currX)
        .attr('y2', (d: any) => updatedNodes[d.target.id].currY)
        .attr('stroke', (d: any) => {
           const isActiveEdge = activeId !== null && (d.source.id === activeId || d.target.id === activeId);
           if (isActiveEdge) return '#378ADD'; // highlight color

           const isNotActiveEdge = activeId !== null && !isActiveEdge;
           if (isNotActiveEdge) return '#1a1a1a'; // dim

           const c1 = updatedNodes[d.source.id].color;
           const c2 = updatedNodes[d.target.id].color;
           if (c1 === '#1D9E75' || c2 === '#1D9E75') return '#11654a';
           if (c1 === '#EF9F27' || c2 === '#EF9F27') return '#A56515';
           if (c1 === '#C5A059' || c2 === '#C5A059') return '#7A6235';
           if (c1 === '#378ADD' || c2 === '#378ADD') return '#1C4B7A';
           return '#222';
        })
        .attr('stroke-width', (d: any) => {
           const isActiveEdge = activeId !== null && (d.source.id === activeId || d.target.id === activeId);
           if (isActiveEdge) return 1.5;

           const isNotActiveEdge = activeId !== null && !isActiveEdge;
           if (isNotActiveEdge) return 0.2;

           const c1 = updatedNodes[d.source.id].color;
           const c2 = updatedNodes[d.target.id].color;
           return (c1 === '#1D9E75' || c2 === '#1D9E75') ? 0.6 : 0.2;
        })
        .attr('opacity', (d: any) => {
           const isActiveEdge = activeId !== null && (d.source.id === activeId || d.target.id === activeId);
           if (isActiveEdge) return 0.8 + Math.sin(time * 40) * 0.2;

           const isNotActiveEdge = activeId !== null && !isActiveEdge;
           if (isNotActiveEdge) return 0.1;

           const c1 = updatedNodes[d.source.id].color;
           const c2 = updatedNodes[d.target.id].color;
           return (c1 === '#1D9E75' || c2 === '#1D9E75') ? 0.8 : 0.3;
        });

      // Update nodes
      g.selectAll('circle')
        .data(updatedNodes)
        .join('circle')
        .attr('cx', (d: any) => d.currX)
        .attr('cy', (d: any) => d.currY)
        .attr('r', (d: any) => {
           const isActiveNode = activeId !== null && d.id === activeId;
           if (isActiveNode) return d.size * (1 + d.z * 0.3) * 1.5 + (Math.sin(time * 30 + d.id) * 1.5);
           return d.size * (1 + d.z * 0.3) + (Math.sin(time * 20 + d.id) * 0.5 * (d.opacity >= 0.8 ? 1 : 0));
        })
        .attr('fill', (d: any) => {
           const isActiveNode = activeId !== null && d.id === activeId;
           return isActiveNode ? '#fff' : d.color;
        })
        .attr('opacity', (d: any) => {
           if (activeId !== null) {
              const isActiveNode = d.id === activeId;
              const isConnectedNode = edges.some(e => 
                  (e.source.id === activeId && e.target.id === d.id) || 
                  (e.target.id === activeId && e.source.id === d.id)
              );
              if (isActiveNode || isConnectedNode) return Math.min(1, (d.opacity + d.z * 0.2) * 1.5);
              return Math.max(0.05, (d.opacity + d.z * 0.2) * 0.2);
           }
           return d.opacity + d.z * 0.2;
        })
        .style('cursor', 'pointer')
        .on('click', (event: any, d: any) => {
          activeNodeIdRef.current = activeNodeIdRef.current === d.id ? null : d.id;
          onNodeClick(d.nodeData);
        });

      // Update text
      g.selectAll('text')
        .data(updatedNodes)
        .join('text')
        .text((d: any) => d.nodeData.label)
        .attr('x', (d: any) => d.currX + 8)
        .attr('y', (d: any) => d.currY + 4)
        .attr('fill', '#666')
        .attr('font-size', '10px')
        .attr('font-family', 'ui-sans-serif, system-ui')
        .attr('opacity', (d: any) => {
           if (activeId !== null) {
              const isActiveNode = d.id === activeId;
              const isConnectedNode = edges.some(e => 
                  (e.source.id === activeId && e.target.id === d.id) || 
                  (e.target.id === activeId && e.source.id === d.id)
              );
              return isActiveNode || isConnectedNode ? 1 : 0.1;
           }
           return 0.8;
        })
        .style('pointer-events', 'none');

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [nodes, edges, onNodeClick]);

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width * 2 - 1; 
    const y = (e.clientY - rect.top) / rect.height * 2 - 1;
    mousePos.current = { x, y };
  };

  return (
    <div 
      ref={containerRef} 
      onMouseMove={handleMouseMove}
      className="relative w-full h-full bg-[#121212] overflow-hidden rounded-xl border border-[#2a2a2a] flex-1 flex items-center justify-center shadow-inner cursor-move"
    >
      <svg ref={svgRef} viewBox="-400 -300 800 600" className="w-full h-full">
        <g ref={gRef}></g>
      </svg>
      {/* Source Showcase Overlay */}
      <div className="absolute bottom-0 left-0 right-0 h-10 border-t border-[#1e1e1e] bg-[#0A0D14]/80 backdrop-blur-md flex items-center px-4 overflow-hidden pointer-events-none">
        <div className="flex items-center gap-2 whitespace-nowrap text-[10px] text-[#378ADD] font-mono tracking-wider animate-[pulse_4s_ease-in-out_infinite]">
          <span className="text-[#378ADD] font-bold">▶ </span>
          WhaleWisdom 13F: Druckenmiller positions — Web Unlocker verified
        </div>
      </div>
    </div>
  );
}
