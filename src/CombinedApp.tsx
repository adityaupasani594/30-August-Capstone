import { useEffect, useMemo, useState } from 'react';
import { ArrowRight, BrainCircuit, Play, RefreshCw } from 'lucide-react';
import { api } from '../capstone-optimization/frontend/src/services/api';
import type { NetworkResponse, OptimizationResponse } from '../capstone-optimization/frontend/src/types';

type NodePoint = { id: string; x: number; y: number };

const points: NodePoint[] = [
  { id: 'V1', x: 72, y: 75 }, { id: 'V2', x: 225, y: 75 },
  { id: 'V3', x: 378, y: 75 }, { id: 'V8', x: 531, y: 75 },
  { id: 'V4', x: 72, y: 225 }, { id: 'V5', x: 225, y: 225 },
  { id: 'V6', x: 378, y: 225 }, { id: 'V7', x: 531, y: 225 },
];

const edgeOrder = [
  ['V1', 'V2'], ['V1', 'V4'], ['V1', 'V5'], ['V2', 'V3'], ['V2', 'V5'],
  ['V3', 'V8'], ['V4', 'V5'], ['V5', 'V6'], ['V6', 'V2'], ['V6', 'V7'], ['V7', 'V3'],
] as const;

const fallbackNetwork: NetworkResponse = {
  nodes: points.map((point, index) => ({ id: point.id, label: point.id, initial_cycle_time: [55, 75, 70, 55, 60, 80, 75, 60][index] })),
  edges: edgeOrder.map(([source, target]) => ({ id: `${source}→${target}`, source, target, weight: 1, capacity: 3000, speed: 60, lanes: 3, length: 2, road_type: 'Arterial', threshold: 1800, is_reference: false })),
  edge_features: {}, predictions: Object.fromEntries(edgeOrder.map(([source, target], index) => [`${source}→${target}`, [630, 520, 650, 602, 717, 494, 676, 792, 379, 586, 525][index]])), thresholds: {},
};

export default function CombinedApp() {
  const [network, setNetwork] = useState<NetworkResponse>(fallbackNetwork);
  const [pso, setPso] = useState<OptimizationResponse | null>(null);
  const [qaoa, setQaoa] = useState<OptimizationResponse | null>(null);
  const [running, setRunning] = useState<'pso' | 'qaoa' | null>(null);
  const [error, setError] = useState('');
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [sliderValue, setSliderValue] = useState<number>(0);

  useEffect(() => { api.getNetwork('single8').then(setNetwork).catch(() => setError('Showing the local 8-node network. Start FastAPI for live optimization.')); }, []);

  useEffect(() => {
    if (selectedEdgeId) {
      setSliderValue(network.predictions[selectedEdgeId] ?? 0);
    }
  }, [selectedEdgeId, network.predictions]);

  const run = async (algorithm: 'pso' | 'qaoa') => {
    setRunning(algorithm); setError('');
    const capacities = Object.fromEntries(network.edges.map(edge => [edge.id, edge.capacity]));
    try {
      const response = algorithm === 'pso'
        ? await api.runOptimization({ network_type: 'single8', capacities, predictions: network.predictions, force: true })
        : await api.runQaoaOptimization({ network_type: 'single8', capacities, predictions: network.predictions });
      
      if (algorithm === 'pso') {
        setPso(response);
      } else {
        setQaoa(response);
      }

      const updatedTraffic = response.after || response.optimized_congestion;
      if (updatedTraffic) {
        setNetwork(prev => ({
          ...prev,
          predictions: {
            ...prev.predictions,
            ...updatedTraffic,
          }
        }));
      }
    } catch { setError(`${algorithm.toUpperCase()} could not run. Check that the FastAPI backend is running.`); }
    finally { setRunning(null); }
  };

  const handleEdgePredictionChange = async (edgeId: string, newValue: number) => {
    if (!edgeId) return;
    const clamped = Math.max(0, Number(newValue) || 0);
    setSliderValue(clamped);

    setNetwork(prev => {
      const nextPredictions = { ...prev.predictions, [edgeId]: clamped };
      return { ...prev, predictions: nextPredictions };
    });

    try {
      const result = await api.stqgcnPredict({
        predictions: { ...network.predictions, [edgeId]: clamped },
        changed_edge: edgeId,
        new_value: clamped,
        network_type: 'single8',
        edges: network.edges.map(edge => ({ id: edge.id, source: edge.source, target: edge.target, capacity: edge.capacity })),
      });
      setNetwork(prev => ({ ...prev, predictions: result.predictions }));
      setPso(null);
      setQaoa(null);
    } catch {
      setError('STQGCN prediction update failed.');
    }
  };

  const active = qaoa || pso;
  const timings = useMemo(() => network.nodes.map(node => {
    const incoming = network.edges.filter(edge => edge.target === node.id);
    const old = node.initial_cycle_time;
    const after = active?.cycle_times?.[node.id]?.new ?? old;
    const green = active?.green_times ? Object.entries(active.green_times).filter(([id]) => incoming.some(edge => edge.id === id)).reduce((sum, [, value]) => sum + value.new, 0) : 0;
    return { id: node.id, old, after, green: Math.round(green || (old * .55)) };
  }), [network, pso, qaoa, active]);

  return <div className="network-app">
    <main className="network-main">
      <div className="network-title"><div><p>NETWORK TOPOLOGY</p><h1>Mumbai single network</h1><span>Traffic forecast → signal optimization</span></div><button className="refresh-network" onClick={() => window.location.reload()}><RefreshCw size={15} /> Refresh</button></div>
      {error && <div className="network-error">{error}</div>}
      <section className="network-layout"><div className="network-map"><div className="map-label">DIRECTED ROAD GRAPH</div><svg viewBox="0 0 603 300" aria-label="Mumbai 8 node directed road graph"><defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#78d9b5" /></marker></defs><rect width="603" height="300" fill="#f8f8f6" />{edgeOrder.map(([source, target]) => { const a = points.find(point => point.id === source)!; const b = points.find(point => point.id === target)!; const dx = b.x - a.x; const dy = b.y - a.y; const length = Math.max(1, Math.hypot(dx, dy)); const startX = a.x + dx / length * 14; const startY = a.y + dy / length * 14; const endX = b.x - dx / length * 14; const endY = b.y - dy / length * 14; const edgeId = `${source}→${target}`; const value = network.predictions[edgeId] ?? 0; const threshold = network.edges.find(edge => edge.id === edgeId)?.threshold ?? 1800; const edgeClass = value > threshold ? 'draw-critical' : value > threshold * 0.85 ? 'draw-watch' : ''; return <line key={`${source}-${target}`} x1={startX} y1={startY} x2={endX} y2={endY} markerEnd="url(#arrow)" className={`draw-edge ${edgeClass}`} onClick={() => setSelectedEdgeId(edgeId)} style={{ cursor: 'pointer' }} />; })}{points.map(point => <g key={point.id}><circle cx={point.x} cy={point.y} r="18" className="draw-node" /><text x={point.x} y={point.y + 5} textAnchor="middle">{point.id}</text></g>)}</svg>{selectedEdgeId && <div className="network-edge-panel"><div className="network-edge-header"><span>{selectedEdgeId}</span><button onClick={() => setSelectedEdgeId(null)}>Close</button></div><input type="range" min={0} max={2200} step={10} value={sliderValue} onChange={(e) => { const next = Number(e.target.value); handleEdgePredictionChange(selectedEdgeId, next); }} /><div className="network-edge-value"><strong>{Math.round(sliderValue)}</strong><small>veh/hr</small></div></div>}</div><aside className="edge-list"><div className="map-label">EDGE REGISTER</div><div className="edge-columns"><span>EDGE</span><span>FROM → TO</span><span>TRAFFIC</span></div>{edgeOrder.map(([source, target], index) => { const edgeId = `${source}→${target}`; const value = network.predictions[edgeId] ?? 0; return <div className="edge-item" key={`${source}-${target}`}><b>E{index}</b><span>{source} <ArrowRight size={13} /> {target}</span><strong>{Math.round(value)} veh/hr</strong></div>; })}</aside></section>
      <section className="optimizer-row"><div className="optimizer-intro"><p>DECISION LAYER</p><h2>Optimize the exact network above</h2><span>Both algorithms consume the same 8-node forecast and return new signal timings.</span></div><button className="pso-action" disabled={!!running} onClick={() => run('pso')}><Play size={15} fill="currentColor" />{running === 'pso' ? 'Running PSO…' : 'Run PSO'}</button><button className="qaoa-action" disabled={!!running} onClick={() => run('qaoa')}><BrainCircuit size={16} />{running === 'qaoa' ? 'Running QAOA…' : 'Run QAOA'}</button></section>
      <section className="timing-panel"><div className="timing-heading"><div><p>SIGNAL PLAN</p><h2>Before and after optimization</h2></div><span>{active ? qaoa ? 'QAOA result' : 'PSO result' : 'Run an optimizer to populate after times'}</span></div><div className="timing-table"><div className="timing-line timing-head"><span>JUNCTION</span><span>BEFORE CYCLE</span><span>AFTER CYCLE</span><span>GREEN TIME</span></div>{timings.map(row => <div className="timing-line" key={row.id}><b>{row.id}</b><span>{row.old}s</span><strong className={active ? 'timing-changed' : ''}>{row.after}s</strong><span>{row.green}s</span></div>)}</div></section>
    </main>
  </div>;
}
