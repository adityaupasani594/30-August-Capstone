import { Handle, Position } from '@xyflow/react';

interface TrafficNodeProps {
  data: {
    label: string;
    isOptimized?: boolean;
  };
}

export default function TrafficNode({ data }: TrafficNodeProps) {
  // Use a slightly different style if we are in the 'after' graph
  const isOpt = data.isOptimized;
  const bg = isOpt ? '#14532d' : '#1e293b';
  const border = isOpt ? '#22c55e' : '#475569';

  return (
    <div
      style={{
        background: bg,
        color: '#f1f5f9',
        border: `2px solid ${border}`,
        borderRadius: 8,
        fontWeight: 700,
        fontSize: 16,
        width: 48,
        height: 48,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <Handle type="target" position={Position.Top} id="top" style={{ visibility: 'hidden' }} />
      <Handle type="target" position={Position.Left} id="left" style={{ visibility: 'hidden' }} />
      <Handle type="target" position={Position.Bottom} id="bottom" style={{ visibility: 'hidden' }} />
      <Handle type="target" position={Position.Right} id="right" style={{ visibility: 'hidden' }} />
      
      <Handle type="source" position={Position.Top} id="top-src" style={{ visibility: 'hidden' }} />
      <Handle type="source" position={Position.Left} id="left-src" style={{ visibility: 'hidden' }} />
      <Handle type="source" position={Position.Bottom} id="bottom-src" style={{ visibility: 'hidden' }} />
      <Handle type="source" position={Position.Right} id="right-src" style={{ visibility: 'hidden' }} />
      
      {data.label}
    </div>
  );
}
