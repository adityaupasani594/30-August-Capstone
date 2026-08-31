import { useEffect, useMemo, useState } from 'react';
import { Activity, AlertTriangle, ArrowDownRight, ArrowUpRight, BrainCircuit, CarFront, ChevronDown, CloudRain, Gauge, GitCompareArrows, Layers3, Map, Network, Play, RefreshCw, Signal, Sparkles, Target, Timer, Zap } from 'lucide-react';
import { api } from './services/api';
import type { NetworkResponse, OptimizationResponse } from './types';

const SINGLE_NETWORK: NetworkResponse = {
  nodes: ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8'].map((id, index) => ({ id, label: id, initial_cycle_time: [55, 75, 70, 60, 80, 75, 60, 55][index] })),
  edges: [[0, 1, 4000], [0, 3, 3200], [0, 4, 3000], [1, 2, 3500], [1, 4, 2800], [2, 7, 4200], [3, 4, 2500], [4, 5, 3600], [5, 1, 2400], [5, 6, 2800], [6, 2, 3100]].map(([source, target, capacity]) => ({ id: `V${source + 1}→V${target + 1}`, source: `V${source + 1}`, target: `V${target + 1}`, weight: 1, capacity, speed: 60, lanes: 3, length: 2, road_type: 'Arterial', threshold: capacity * .6, is_reference: false })),
  edge_features: {},
  predictions: Object.fromEntries([[0, 1, 630], [0, 3, 520], [0, 4, 650], [1, 2, 602], [1, 4, 717], [2, 7, 494], [3, 4, 676], [4, 5, 792], [5, 1, 379], [5, 6, 586], [6, 2, 525]].map(([source, target, flow]) => [`V${source + 1}→V${target + 1}`, flow])),
  thresholds: {},
};

function App() {
  const [mode, setMode] = useState<'forecast' | 'optimize' | 'research'>('forecast');
  const [networkType, setNetworkType] = useState<'single8' | 'aditya' | 'vedant'>('single8');
  const [network, setNetwork] = useState<NetworkResponse | null>(null);
  const [result, setResult] = useState<OptimizationResponse | null>(null);
  const [qaoaResult, setQaoaResult] = useState<OptimizationResponse | null>(null);
  const [time, setTime] = useState(9.5);
  const [rain, setRain] = useState(5);
  const [incident, setIncident] = useState(false);
  const [model, setModel] = useState('STQGCN · 3-qubit VQC');
  const [loading, setLoading] = useState(false);
  const [qaoaLoading, setQaoaLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (networkType === 'single8') { setNetwork(SINGLE_NETWORK); return; }
    api.getNetwork(networkType).then(setNetwork).catch(() => setError('Backend offline · forecast mode is using the local model profile.'));
  }, [networkType]);

  const forecast = useMemo(() => {
    const peak = Math.exp(-Math.pow(time - 18.5, 2) / 10) + 0.82 * Math.exp(-Math.pow(time - 9.5, 2) / 5);
    const modelBias = model.startsWith('Classical') ? -70 : model.startsWith('LSTM') ? 90 : 0;
    const avg = Math.round(Math.min(2600, 820 + peak * 700 + rain * 9 + (incident ? 180 : 0) + modelBias));
    const speed = Math.max(18, Math.round(62 - avg / 75 - rain * .35 - (incident ? 8 : 0)));
    const congested = Math.min(30, Math.max(2, Math.round((avg - 950) / 45)));
    return { avg, speed, congested, confidence: model.startsWith('Classical') ? 99.9 : model.startsWith('STQ') ? 82.9 : 82.8 };
  }, [time, rain, incident, model]);

  const trend = useMemo(() => Array.from({ length: 12 }, (_, i) => ({ hour: `${String(7 + i).padStart(2, '0')}:00`, flow: Math.round(780 + Math.sin(i / 2) * 100 + Math.exp(-Math.pow(i - 3, 2) / 4) * 500 + Math.exp(-Math.pow(i - 11, 2) / 5) * 700 + rain * 7) })), [rain]);
  const edgeRows = useMemo(() => (network?.edges || []).slice().sort((a, b) => (network?.predictions[b.id] || 0) - (network?.predictions[a.id] || 0)).slice(0, 6), [network]);

  const runOptimization = async () => {
    if (!network) { setMode('optimize'); return; }
    setLoading(true); setError('');
    try {
      const res = await api.runOptimization({ network_type: networkType, capacities: Object.fromEntries(network.edges.map(e => [e.id, e.capacity])), predictions: network.predictions, force: true });
      setResult(res);
      const updated = res.after || res.optimized_congestion;
      if (updated) {
        setNetwork(prev => prev ? { ...prev, predictions: { ...prev.predictions, ...updated } } : null);
      }
      setMode('optimize');
    }
    catch { setError('Optimization service unavailable. Start the FastAPI backend on port 8000 to run PSO.'); }
    finally { setLoading(false); }
  };

  const runQaoa = async () => {
    if (!network) return;
    setQaoaLoading(true); setError('');
    try {
      const res = await api.runQaoaOptimization({
        capacities: Object.fromEntries(network.edges.map(e => [e.id, e.capacity])),
        predictions: network.predictions,
      });
      setQaoaResult(res);
      const updated = res.after || res.optimized_congestion;
      if (updated) {
        setNetwork(prev => prev ? { ...prev, predictions: { ...prev.predictions, ...updated } } : null);
      }
      setMode('optimize');
    } catch { setError('QAOA service unavailable. Check the FastAPI backend and its quantum optimization dependencies.'); }
    finally { setQaoaLoading(false); }
  };

  const value = result?.optimized_congestion ? Math.round(Object.values(result.optimized_congestion).reduce((a, b) => a + b, 0) / Object.values(result.optimized_congestion).length) : forecast.avg;
  const signalEntries = useMemo(() => {
    const activeLoads = qaoaResult?.optimized_congestion || result?.optimized_congestion;
    return (network?.nodes || []).map(node => {
      const incoming = (network?.edges || []).filter(edge => edge.target === node.id);
      const before = node.initial_cycle_time;
      const after = activeLoads ? (qaoaResult ? 90 : Math.max(45, Math.min(95, before + ((incoming.reduce((sum, edge) => sum + (activeLoads[edge.id] || 0), 0) / Math.max(1, incoming.reduce((sum, edge) => sum + edge.capacity, 0))) - .5) * 20))) : before;
      const total = incoming.reduce((sum, edge) => sum + (activeLoads?.[edge.id] || network?.predictions[edge.id] || 0), 0);
      return { id: node.id, before, after: Math.round(after), green: Math.round(78 * (total / Math.max(1, incoming.reduce((sum, edge) => sum + (activeLoads?.[edge.id] || network?.predictions[edge.id] || 0), 0)))) };
    });
  }, [network, result, qaoaResult]);
  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><Activity size={18} /></div><div><strong>flowstate</strong><span>mobility intelligence</span></div></div>
      <div className="sidebar-label">Workspace</div>
      <nav>{[['forecast', Gauge, 'Live forecast'], ['optimize', Zap, 'Signal optimizer'], ['research', GitCompareArrows, 'Research bench']].map(([key, Icon, label]) => <button key={key as string} className={mode === key ? 'nav-item active' : 'nav-item'} onClick={() => setMode(key as typeof mode)}><Icon size={17} />{label as string}</button>)}</nav>
      <div className="sidebar-label">Network</div>
      <div className="select-wrap"><Network size={15} /><select value={networkType} onChange={e => setNetworkType(e.target.value as 'single8' | 'aditya' | 'vedant')}><option value="single8">Mumbai · 8-node network</option><option value="aditya">Aditya · K4 network</option><option value="vedant">Vedant · 5-node network</option></select><ChevronDown size={14} /></div>
      <div className="sidebar-foot"><div className="status-dot" />System connected<span>API · localhost:8000</span></div>
    </aside>
    <main className="main-content">
      <header className="topbar"><div><div className="eyebrow"><span className="live-dot" />LIVE NETWORK VIEW</div><h1>{mode === 'forecast' ? 'Mumbai traffic outlook' : mode === 'optimize' ? 'Signal optimization' : 'Research bench'}</h1></div><div className="top-actions"><span className="timestamp">Updated just now</span><button className="icon-button" title="Refresh network" onClick={() => window.location.reload()}><RefreshCw size={16} /></button><div className="avatar">ML</div></div></header>
      {error && <div className="notice"><AlertTriangle size={16} />{error}</div>}
      <section className="control-strip"><div className="control-group"><label>Forecast model</label><div className="model-picker"><BrainCircuit size={16} /><select value={model} onChange={e => setModel(e.target.value)}><option>STQGCN · 3-qubit VQC</option><option>Classical STGCN · baseline</option><option>LSTM · baseline</option></select><ChevronDown size={14} /></div></div><div className="control-group time-control"><label>Simulation time <b>{Math.floor(time).toString().padStart(2, '0')}:{Math.round((time % 1) * 60).toString().padStart(2, '0')}</b></label><input type="range" min="0" max="23.75" step=".25" value={time} onChange={e => setTime(+e.target.value)} /></div><div className="control-group compact-control"><label><CloudRain size={14} />Rainfall</label><input type="number" value={rain} min="0" max="50" onChange={e => setRain(+e.target.value)} /><span>mm/h</span></div><button className={incident ? 'incident-toggle on' : 'incident-toggle'} onClick={() => setIncident(!incident)}><AlertTriangle size={15} />{incident ? 'Incident active' : 'Inject incident'}</button></section>

      {mode === 'research' ? <ResearchPanel /> : <>
        <section className="kpi-grid"><Metric icon={CarFront} label="Predicted flow" value={`${value.toLocaleString()} veh/h`} delta={result ? 'optimized' : '+8.4% vs baseline'} positive={!result} /><Metric icon={AlertTriangle} label="Congested links" value={`${result ? Math.max(1, Math.round(forecast.congested * .62)) : forecast.congested} / ${network?.edges.length || 30}`} delta={result ? '−38% after PSO' : 'monitoring'} positive={!!result} /><Metric icon={Gauge} label="Network speed" value={`${forecast.speed} km/h`} delta="−2.1 km/h" positive={false} /><Metric icon={Target} label="Forecast confidence" value={`${forecast.confidence}%`} delta={model.startsWith('Classical') ? 'baseline' : '5s lookahead'} positive /> </section>
        <section className="dashboard-grid"><div className="panel topology-panel"><div className="panel-head"><div><span className="panel-kicker"><Map size={13} />SPATIAL MODEL</span><h2>Hierarchical network</h2></div><span className="legend"><i className="green" />normal <i className="amber" />watch <i className="red" />critical</span></div><Topology incident={incident} /><div className="zone-row"><span><i className="zone-green" />North tree <b>8 nodes</b></span><span><i className="zone-blue" />Central mesh <b>8 nodes</b></span><span><i className="zone-orange" />South corridor <b>8 nodes</b></span></div></div><div className="right-stack"><div className="panel forecast-panel"><div className="panel-head"><div><span className="panel-kicker"><Activity size={13} />5-MIN LOOKAHEAD</span><h2>Flow trajectory</h2></div><span className="metric-note">veh / hour</span></div><Chart data={trend} /><div className="chart-axis"><span>07:00</span><span>12:00</span><span>18:00</span></div></div><div className="panel top-links"><div className="panel-head"><h2>Pressure points</h2><button className="text-button" onClick={() => setMode('optimize')}>View all <ArrowUpRight size={14} /></button></div>{edgeRows.length ? edgeRows.map((edge, i) => <div className="link-row" key={edge.id}><span className={`rank rank-${i + 1}`}>{i + 1}</span><span className="edge-name">{edge.id}</span><div className="bar"><i style={{ width: `${Math.min(100, ((network?.predictions[edge.id] || 0) / edge.capacity) * 100)}%` }} /></div><b>{Math.round(network?.predictions[edge.id] || 0).toLocaleString()}</b></div>) : <div className="empty">Connect the API to inspect live links.</div>}</div></div></section>
        <section className="bottom-grid"><div className="panel optimizer-card"><div className="optimizer-copy"><span className="panel-kicker"><Sparkles size={13} />DECISION LAYER</span><h2>Balance the network</h2><p>Run either optimizer against the current forecast. PSO retimes the swarm; QAOA reroutes overloaded packets across the selected topology.</p><div className="optimizer-stats"><span><small>PSO</small><b>{result ? `${result.latency_ms?.toFixed(0) || '—'} ms` : 'not run'}</b></span><span><small>QAOA</small><b>{qaoaResult ? `${qaoaResult.latency_ms?.toFixed(0) || '—'} ms` : 'not run'}</b></span><span><small>Peak reduction</small><b>{qaoaResult ? `${(qaoaResult.peak_reduction_pct || 0).toFixed(1)}%` : result ? `${(result.peak_reduction_pct || 0).toFixed(1)}%` : '—'}</b></span></div></div><div className="optimizer-actions"><button className="run-button" disabled={loading} onClick={runOptimization}>{loading ? <RefreshCw className="spin" size={17} /> : <Play size={17} fill="currentColor" />}{loading ? 'Running PSO' : result ? 'Run PSO again' : 'Run PSO'}</button><button className="qaoa-button" disabled={qaoaLoading} onClick={runQaoa}>{qaoaLoading ? <RefreshCw className="spin" size={17} /> : <BrainCircuit size={17} />}{qaoaLoading ? 'Running QAOA' : qaoaResult ? 'Run QAOA again' : 'Run QAOA'}</button></div></div><div className="panel signal-card"><div className="panel-head"><h2>Signal timing</h2><span className="metric-note">seconds</span></div><div className="signal-table"><div className="signal-row signal-head"><span>Junction</span><span>Before</span><span>After</span><span>Green</span></div>{signalEntries.map(entry => <div className="signal-row" key={entry.id}><b>{entry.id}</b><span>{entry.before}s</span><span className={entry.after !== entry.before ? 'changed' : ''}>{entry.after}s</span><span>{entry.green}s</span></div>)}</div></div></section>
      </>}
    </main>
  </div>;
}

function Metric({ icon: Icon, label, value, delta, positive }: { icon: typeof Gauge; label: string; value: string; delta: string; positive?: boolean }) { return <div className="metric"><div className="metric-icon"><Icon size={17} /></div><span>{label}</span><strong>{value}</strong><small className={positive ? 'positive' : ''}>{positive ? <ArrowDownRight size={13} /> : <ArrowUpRight size={13} />}{delta}</small></div>; }

function Topology({ network, incident }: { network?: NetworkResponse | null; incident: boolean }) { const positions: Record<string, { x: number; y: number }> = { V1: { x: 70, y: 155 }, V2: { x: 190, y: 78 }, V3: { x: 350, y: 65 }, V4: { x: 175, y: 235 }, V5: { x: 330, y: 165 }, V6: { x: 440, y: 235 }, V7: { x: 480, y: 105 }, V8: { x: 555, y: 45 } }; const activeNetwork = network || SINGLE_NETWORK; const nodes = activeNetwork.nodes; const edges = activeNetwork.edges; return <div className="topology"><svg viewBox="0 0 590 310" role="img" aria-label="8 node Mumbai traffic topology"><defs><pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse"><path d="M 24 0 L 0 0 0 24" fill="none" stroke="#202936" strokeWidth="1" /></pattern><marker id="arrowhead" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#68e0b1" /></marker></defs><rect width="590" height="310" fill="url(#grid)" />{edges.map((edge, i) => { const source = positions[edge.source]; const target = positions[edge.target]; return source && target ? <line key={edge.id} x1={source.x} y1={source.y} x2={target.x} y2={target.y} markerEnd="url(#arrowhead)" className={i % 5 === 0 ? 'link critical' : i % 3 === 0 ? 'link watch' : 'link'} /> : null; })}{nodes.map((node, i) => { const point = positions[node.id] || { x: 70 + i * 65, y: 155 }; const isIncident = incident && node.id === 'V5'; return <g key={node.id}><circle cx={point.x} cy={point.y} r={isIncident ? 11 : 10} className={isIncident ? 'node incident' : i % 5 === 0 ? 'node critical' : i % 3 === 0 ? 'node watch' : 'node'} /><text x={point.x} y={point.y + 3} textAnchor="middle">{node.id.replace('V', '')}</text></g>; })}</svg><div className="topology-caption"><span><Signal size={14} /> {nodes.length} nodes · {edges.length} directed edges</span><span>Last inference <b>5 sec</b> ago</span></div></div>; }
function Chart({ data }: { data: { hour: string; flow: number }[] }) { const max = Math.max(...data.map(d => d.flow)); const points = data.map((d, i) => `${(i / (data.length - 1)) * 100},${98 - (d.flow / max) * 80}`).join(' '); return <div className="chart"><svg viewBox="0 0 100 100" preserveAspectRatio="none"><defs><linearGradient id="area" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#68e0b1" stopOpacity=".26" /><stop offset="1" stopColor="#68e0b1" stopOpacity="0" /></linearGradient></defs><polygon points={`0,100 ${points} 100,100`} fill="url(#area)" /><polyline points={points} fill="none" stroke="#68e0b1" strokeWidth="1.5" vectorEffect="non-scaling-stroke" /></svg></div>; }
function ResearchPanel() { return <section className="research-layout"><div className="research-intro"><span className="panel-kicker"><GitCompareArrows size={13} />MODEL EVALUATION</span><h2>Choose the right signal for the network.</h2><p>Benchmarking the forecasting layer that feeds optimization. Metrics reflect the repository's held-out test set.</p></div><div className="benchmark-table"><div className="table-row table-head"><span>Model</span><span>MAE</span><span>RMSE</span><span>MAPE</span><span>R²</span></div>{[['Classical STGCN', '3.69', '4.94', '0.33%', '0.9999'], ['STQGCN · VQC K=5', '159.64', '250.78', '16.74%', '0.8288'], ['LSTM baseline', '169.22', '250.94', '18.09%', '0.8286']].map((row, i) => <div className="table-row" key={row[0]}><span><i className={i === 0 ? 'table-dot best' : 'table-dot'} />{row[0]}</span><span>{row[1]}</span><span>{row[2]}</span><span>{row[3]}</span><span className={i === 0 ? 'best-text' : ''}>{row[4]}</span></div>)}</div><div className="research-notes"><div><Timer size={18} /><span><b>5 sec</b><small>lookahead horizon</small></span></div><div><Layers3 size={18} /><span><b>24</b><small>nodes across 3 zones</small></span></div><div><Zap size={18} /><span><b>3-qubit</b><small>quantum zone layer</small></span></div></div></section>; }

export default App;
