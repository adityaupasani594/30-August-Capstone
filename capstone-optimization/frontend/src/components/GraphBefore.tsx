/**
 * components/GraphBefore.tsx
 * ==========================
 * React Flow graph showing the STQGCN-predicted (pre-optimization) edge
 * congestion values.
 */

import { useMemo } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MarkerType,
  type Node as RFNode,
  type Edge as RFEdge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import TrafficNode from './TrafficNode';

const nodeTypes = { traffic: TrafficNode };

// ---------------------------------------------------------------------------
// Layout — fixed positions for the 5-node graph
// ---------------------------------------------------------------------------
const NODE_POSITIONS: Record<string, { x: number; y: number }> = {
  A: { x: 290, y: 40 },
  B: { x: 60,  y: 200 },
  C: { x: 290, y: 280 },
  D: { x: 520, y: 200 },
  E: { x: 520, y: 400 },
};

// Edge definitions — with explicit handles to prevent overlap and missing arrows
const EDGE_DEFS = [
  { source: 'A', target: 'B', isRef: false, srcHandle: 'left-src', tgtHandle: 'top' },
  { source: 'A', target: 'C', isRef: false, srcHandle: 'bottom-src', tgtHandle: 'top' },
  { source: 'A', target: 'D', isRef: false, srcHandle: 'right-src', tgtHandle: 'top' },
  { source: 'B', target: 'D', isRef: false, srcHandle: 'right-src', tgtHandle: 'left' },
  { source: 'C', target: 'D', isRef: false, srcHandle: 'right-src', tgtHandle: 'bottom' },
  { source: 'D', target: 'E', isRef: false, srcHandle: 'bottom-src', tgtHandle: 'top' },
  { source: 'E', target: 'B', isRef: false,  srcHandle: 'left-src', tgtHandle: 'bottom' },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Map congestion (veh/hr) to a traffic-severity colour. */
function edgeColour(congestion: number, isRef: boolean): string {
  if (isRef)              return '#9ca3af';
  if (congestion < 1000)  return '#22c55e';
  if (congestion <= 1500) return '#f97316';
  return '#ef4444';
}

/** Build React Flow nodes from the fixed layout. */
function buildNodes(): RFNode[] {
  return Object.entries(NODE_POSITIONS).map(([id, position]) => ({
    id,
    type: 'traffic',
    position,
    data: { label: id, isOptimized: false },
  }));
}

/** Build React Flow edges with congestion labels and colour. */
function buildEdges(congestion: Record<string, number>): RFEdge[] {
  return EDGE_DEFS.map(({ source, target, isRef, srcHandle, tgtHandle }) => {
    const key  = `${source}\u2192${target}`;
    const cong = congestion[key] ?? 0;
    const col  = edgeColour(cong, isRef);
    const label = isRef
      ? `REF: ${Math.round(cong)} veh/hr`
      : `${Math.round(cong)} veh/hr`;

    return {
      id: `e-${source}-${target}`,
      source,
      target,
      sourceHandle: srcHandle,
      targetHandle: tgtHandle,
      label,
      type: 'bezier',
      animated: !isRef && cong > 1500,
      style: { stroke: col, strokeWidth: 2.5 },
      labelStyle: { fontSize: 11, fontWeight: 600, fill: col },
      labelBgStyle: { fill: '#0f172a', fillOpacity: 0.85 },
      labelBgPadding: [4, 6] as [number, number],
      markerEnd: { type: MarkerType.ArrowClosed, color: col },
    };
  });
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface GraphBeforeProps {
  /** Map of edge_id → predicted congestion in veh/hr (from STQGCN). */
  congestion: Record<string, number>;
}

export default function GraphBefore({ congestion }: GraphBeforeProps) {
  const nodes = useMemo(() => buildNodes(), []);
  const edges = useMemo(() => buildEdges(congestion), [congestion]);

  return (
    <div style={{ width: '100%', height: 560, background: '#0f172a', borderRadius: 8 }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#1e293b" gap={20} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
