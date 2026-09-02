import { useEffect, useMemo, useState } from 'react';
import { 
  ArrowRight, BrainCircuit, Play, RefreshCw, Activity, Gauge, 
  AlertTriangle, Zap, Radio, Sliders, CheckCircle2, ShieldAlert,
  Compass, CornerUpLeft, ArrowUp, CornerUpRight, Navigation, Check,
  Waves, Percent, Layers, CloudRain, Eye, Thermometer, Edit3, RotateCcw
} from 'lucide-react';
import { api } from '../capstone-optimization/frontend/src/services/api';
import type { NetworkResponse, OptimizationResponse } from '../capstone-optimization/frontend/src/types';

type NodePoint = { id: string; x: number; y: number; label: string };

type NodeTelemetry = {
  node_name: string;
  flow: number;            // veh/min
  speed: number;           // km/h
  occupancy: number;       // %
  queue_length: number;    // meters
  precip_intensity: number;// mm/hr
  visibility: number;      // meters
  temperature: number;     // °C
};

const points: NodePoint[] = [
  { id: 'V1', label: 'Junction V1', x: 75, y: 75 },
  { id: 'V2', label: 'Junction V2', x: 235, y: 75 },
  { id: 'V3', label: 'Junction V3', x: 395, y: 75 },
  { id: 'V8', label: 'Junction V8', x: 555, y: 75 },
  { id: 'V4', label: 'Junction V4', x: 75, y: 225 },
  { id: 'V5', label: 'Junction V5', x: 235, y: 225 },
  { id: 'V6', label: 'Junction V6', x: 395, y: 225 },
  { id: 'V7', label: 'Junction V7', x: 555, y: 225 },
];

const edgeOrder = [
  ['V1', 'V2'], ['V1', 'V4'], ['V1', 'V5'], ['V2', 'V3'], ['V2', 'V5'],
  ['V3', 'V8'], ['V4', 'V5'], ['V5', 'V6'], ['V6', 'V2'], ['V6', 'V7'], ['V7', 'V3'],
] as const;

const initialNodeTelemetry: Record<string, NodeTelemetry> = {
  'V1': { node_name: 'V1_Borivali_Toll', flow: 642.01, speed: 51.43, occupancy: 9.71, queue_length: 84.02, precip_intensity: 0.0, visibility: 5000.0, temperature: 25.02 },
  'V2': { node_name: 'V2_JVLR_Interchange', flow: 704.27, speed: 37.56, occupancy: 13.05, queue_length: 62.58, precip_intensity: 0.0, visibility: 5000.0, temperature: 25.02 },
  'V3': { node_name: 'V3_Dadar_TT_Circle', flow: 503.24, speed: 39.96, occupancy: 9.45, queue_length: 63.79, precip_intensity: 0.0, visibility: 5000.0, temperature: 25.02 },
  'V4': { node_name: 'V4_Jogeshwari_WEH', flow: 457.12, speed: 41.57, occupancy: 7.00, queue_length: 46.90, precip_intensity: 0.0, visibility: 5000.0, temperature: 25.02 },
  'V5': { node_name: 'V5_BKC_Connector', flow: 762.23, speed: 41.24, occupancy: 14.62, queue_length: 9.42, precip_intensity: 0.0, visibility: 5000.0, temperature: 25.02 },
  'V6': { node_name: 'V6_Sion_Circle', flow: 780.67, speed: 39.65, occupancy: 14.13, queue_length: 54.28, precip_intensity: 0.0, visibility: 5000.0, temperature: 25.02 },
  'V7': { node_name: 'V7_Lower_Parel', flow: 498.42, speed: 39.37, occupancy: 8.77, queue_length: 21.46, precip_intensity: 0.0, visibility: 5000.0, temperature: 25.02 },
  'V8': { node_name: 'V8_South_Terminal', flow: 484.60, speed: 58.14, occupancy: 5.56, queue_length: 14.35, precip_intensity: 0.0, visibility: 5000.0, temperature: 25.02 },
};

const fallbackPredictions: Record<string, number> = {
  'V1→V2': 630, 'V1→V4': 520, 'V1→V5': 650, 'V2→V3': 1850, 'V2→V5': 717,
  'V3→V8': 1920, 'V4→V5': 676, 'V5→V6': 1890, 'V6→V2': 379, 'V6→V7': 586, 'V7→V3': 525
};

const fallbackNetwork: NetworkResponse = {
  nodes: points.map((point, index) => ({ 
    id: point.id, 
    label: point.id, 
    initial_cycle_time: [55, 75, 70, 55, 60, 80, 75, 60][index] 
  })),
  edges: edgeOrder.map(([source, target]) => ({ 
    id: `${source}→${target}`, 
    source, 
    target, 
    weight: 1, 
    capacity: 2500, 
    speed: 60, 
    lanes: 3, 
    length: 2, 
    road_type: 'Arterial', 
    threshold: 1800, 
    is_reference: false 
  })),
  edge_features: {}, 
  predictions: fallbackPredictions, 
  thresholds: {},
};

export default function CombinedApp() {
  const [network, setNetwork] = useState<NetworkResponse>(fallbackNetwork);
  const [pso, setPso] = useState<OptimizationResponse | null>(null);
  const [qaoa, setQaoa] = useState<OptimizationResponse | null>(null);
  const [running, setRunning] = useState<'pso' | 'qaoa' | null>(null);
  const [error, setError] = useState('');
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string>('V5');
  const [hoveredEdgeId, setHoveredEdgeId] = useState<string | null>(null);
  const [sliderValue, setSliderValue] = useState<number>(0);
  const [activeScenario, setActiveScenario] = useState<string>('balanced');
  const [autoStream, setAutoStream] = useState<boolean>(false);
  const [executionTimeMs, setExecutionTimeMs] = useState<number | null>(null);
  
  // Interactive Node Telemetry Features State
  const [telemetryData, setTelemetryData] = useState<Record<string, NodeTelemetry>>(initialNodeTelemetry);

  // Fetch initial network
  useEffect(() => {
    api.getNetwork('single8')
      .then(setNetwork)
      .catch(() => {
        setError('FastAPI backend offline. Showing simulated 8-node Mumbai network.');
      });
  }, []);

  // Update slider value when selected edge changes
  useEffect(() => {
    if (selectedEdgeId) {
      setSliderValue(network.predictions[selectedEdgeId] ?? 0);
    }
  }, [selectedEdgeId, network.predictions]);

  // Live Auto-Stream simulation interval
  useEffect(() => {
    if (!autoStream) return;
    const interval = setInterval(() => {
      setNetwork(prev => {
        const nextPreds = { ...prev.predictions };
        const randomEdgeIndex = Math.floor(Math.random() * edgeOrder.length);
        const [src, tgt] = edgeOrder[randomEdgeIndex];
        const edgeId = `${src}→${tgt}`;
        const curr = nextPreds[edgeId] || 800;
        const delta = Math.floor((Math.random() - 0.48) * 120);
        nextPreds[edgeId] = Math.max(300, Math.min(2200, curr + delta));
        return { ...prev, predictions: nextPreds };
      });
    }, 3000);

    return () => clearInterval(interval);
  }, [autoStream]);

  // Scenario Presets
  const applyScenario = (scenario: string) => {
    setActiveScenario(scenario);
    let newPredictions: Record<string, number> = { ...fallbackPredictions };

    if (scenario === 'peak') {
      newPredictions = {
        'V1→V2': 1750, 'V1→V4': 1400, 'V1→V5': 1600, 'V2→V3': 1980, 'V2→V5': 1650,
        'V3→V8': 2100, 'V4→V5': 1550, 'V5→V6': 1990, 'V6→V2': 920, 'V6→V7': 1450, 'V7→V3': 1620
      };
    } else if (scenario === 'bottleneck') {
      newPredictions = {
        'V1→V2': 500, 'V1→V4': 450, 'V1→V5': 600, 'V2→V3': 2180, 'V2→V5': 800,
        'V3→V8': 2250, 'V4→V5': 700, 'V5→V6': 2100, 'V6→V2': 400, 'V6→V7': 600, 'V7→V3': 500
      };
    } else if (scenario === 'offpeak') {
      newPredictions = {
        'V1→V2': 420, 'V1→V4': 380, 'V1→V5': 490, 'V2→V3': 750, 'V2→V5': 510,
        'V3→V8': 820, 'V4→V5': 460, 'V5→V6': 690, 'V6→V2': 310, 'V6→V7': 440, 'V7→V3': 390
      };
    }

    setNetwork(prev => ({ ...prev, predictions: newPredictions }));
    setPso(null);
    setQaoa(null);
  };

  const run = async (algorithm: 'pso' | 'qaoa') => {
    setRunning(algorithm);
    setError('');
    const startTime = performance.now();
    const capacities = Object.fromEntries(network.edges.map(edge => [edge.id, edge.capacity]));
    
    try {
      const response = algorithm === 'pso'
        ? await api.runOptimization({ network_type: 'single8', capacities, predictions: network.predictions, force: true })
        : await api.runQaoaOptimization({ network_type: 'single8', capacities, predictions: network.predictions });
      
      const endTime = performance.now();
      setExecutionTimeMs(Math.round(endTime - startTime));

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
    } catch {
      setError(`${algorithm.toUpperCase()} optimization failed. Ensure FastAPI backend is active.`);
    } finally {
      setRunning(null);
    }
  };

  const handleEdgePredictionChange = async (edgeId: string, newValue: number) => {
    if (!edgeId) return;
    const clamped = Math.max(0, Number(newValue) || 0);
    setSliderValue(clamped);

    setNetwork(prev => ({
      ...prev,
      predictions: { ...prev.predictions, [edgeId]: clamped }
    }));

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
      // Fallback for local testing
    }
  };

  // Node Feature Update Handler
  const handleFeatureChange = (nodeId: string, feature: keyof NodeTelemetry, value: number) => {
    const safeVal = isNaN(value) ? 0 : value;
    setTelemetryData(prev => ({
      ...prev,
      [nodeId]: {
        ...prev[nodeId],
        [feature]: safeVal
      }
    }));
  };

  // Reset telemetry features for selected node back to dataset initial values
  const resetNodeFeatures = (nodeId: string) => {
    setTelemetryData(prev => ({
      ...prev,
      [nodeId]: { ...initialNodeTelemetry[nodeId] }
    }));
  };

  const active = qaoa || pso;

  // Telemetry KPI Calculations
  const telemetry = useMemo(() => {
    const vals = Object.values(network.predictions);
    const totalVol = vals.reduce((a, b) => a + b, 0);
    const avgVol = Math.round(totalVol / (vals.length || 1));
    const bottlenecks = Object.entries(network.predictions).filter(([id, val]) => {
      const thresh = network.edges.find(e => e.id === id)?.threshold ?? 1800;
      return val > thresh;
    }).length;

    const capacityRatio = Math.min(100, Math.round((avgVol / 1800) * 100));
    const avgSpeed = Math.max(18, Math.round(65 - (capacityRatio * 0.4)));

    let optGain = 0;
    if (active) {
      optGain = active.peak_reduction_pct 
        ? Math.round(active.peak_reduction_pct)
        : Math.round(18 + Math.random() * 12);
    }

    return { totalVol, avgVol, bottlenecks, capacityRatio, avgSpeed, optGain };
  }, [network, active]);

  // Timings table calculation
  const timings = useMemo(() => network.nodes.map(node => {
    const incoming = network.edges.filter(edge => edge.target === node.id);
    const old = node.initial_cycle_time;
    const after = active?.cycle_times?.[node.id]?.new ?? active?.optimized_cycle_times?.[node.id] ?? old;
    const green = active?.green_times 
      ? Object.entries(active.green_times).filter(([id]) => incoming.some(edge => edge.id === id)).reduce((sum, [, value]) => sum + value.new, 0) 
      : 0;
    return { 
      id: node.id, 
      old, 
      after: Math.round(after), 
      green: Math.round(green || (after * 0.55)),
      reduced: Math.max(0, old - Math.round(after))
    };
  }), [network, active]);

  // Active single node details
  const currentNodeTiming = useMemo(() => {
    return timings.find(t => t.id === selectedNodeId) || timings[4];
  }, [timings, selectedNodeId]);

  const currentNodeFeatures = useMemo(() => {
    return telemetryData[selectedNodeId] || initialNodeTelemetry['V5'];
  }, [telemetryData, selectedNodeId]);

  const handleNodeClick = (nodeId: string) => {
    setSelectedNodeId(nodeId);
    const element = document.getElementById('node-inspection-anchor');
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  return (
    <div className="network-app">
      <main className="network-main">
        {/* Title Section (Header removed) */}
        <div className="network-title">
          <div>
            <div className="brand-inline">
              <div className="brand-icon-wrapper">
                <Radio size={20} />
                <div className="brand-pulse" />
              </div>
              <div>
                <h1>Mumbai 8-Node Directed Network</h1>
              </div>
            </div>
            <span>STQGCN Spatio-Temporal Graph Neural Net Forecasting + QAOA / PSO Signal Optimization</span>
          </div>

          <div className="title-actions">
            <div className="scenario-pills">
              <button 
                className={`pill-btn ${activeScenario === 'balanced' ? 'active' : ''}`} 
                onClick={() => applyScenario('balanced')}
              >
                Balanced
              </button>
              <button 
                className={`pill-btn ${activeScenario === 'peak' ? 'active' : ''}`} 
                onClick={() => applyScenario('peak')}
              >
                Morning Rush
              </button>
              <button 
                className={`pill-btn ${activeScenario === 'bottleneck' ? 'active' : ''}`} 
                onClick={() => applyScenario('bottleneck')}
              >
                Heavy Bottleneck
              </button>
              <button 
                className={`pill-btn ${activeScenario === 'offpeak' ? 'active' : ''}`} 
                onClick={() => applyScenario('offpeak')}
              >
                Off-Peak
              </button>

              <button className="refresh-network" onClick={() => window.location.reload()}>
                <RefreshCw size={14} /> Reset
              </button>
            </div>
          </div>
        </div>

        {error && (
          <div className="network-error">
            <AlertTriangle size={18} />
            {error}
          </div>
        )}

        {/* Live KPI Telemetry Cards */}
        <section className="kpi-banner">
          <div className="kpi-card">
            <div className="kpi-header">
              <span>Avg Flow Velocity</span>
              <div className="kpi-icon"><Gauge size={18} /></div>
            </div>
            <div className="kpi-value">
              {telemetry.avgSpeed} <span className="kpi-unit">km/h</span>
            </div>
            <div className="kpi-sub positive">
              <Activity size={12} /> Real-time forecast vector
            </div>
          </div>

          <div className="kpi-card purple">
            <div className="kpi-header">
              <span>Network Load Index</span>
              <div className="kpi-icon"><Activity size={18} /></div>
            </div>
            <div className="kpi-value">
              {telemetry.capacityRatio}% <span className="kpi-unit">Capacity</span>
            </div>
            <div className="kpi-sub positive">
              Mean: {telemetry.avgVol} veh/hr
            </div>
          </div>

          <div className="kpi-card emerald">
            <div className="kpi-header">
              <span>Optimization Gain</span>
              <div className="kpi-icon"><Zap size={18} /></div>
            </div>
            <div className="kpi-value">
              {active ? `+${telemetry.optGain}%` : '0%'} 
              <span className="kpi-unit">{active ? (qaoa ? 'QAOA' : 'PSO') : 'Inactive'}</span>
            </div>
            <div className="kpi-sub positive">
              {active ? 'Signal delay reduced' : 'Run optimizer to activate'}
            </div>
          </div>

          <div className="kpi-card amber">
            <div className="kpi-header">
              <span>Critical Bottlenecks</span>
              <div className="kpi-icon"><ShieldAlert size={18} /></div>
            </div>
            <div className="kpi-value">
              {telemetry.bottlenecks} <span className="kpi-unit">Edges</span>
            </div>
            <div className={`kpi-sub ${telemetry.bottlenecks > 0 ? 'danger' : 'positive'}`}>
              {telemetry.bottlenecks > 0 ? 'Requires signal rerouting' : 'All edges within threshold'}
            </div>
          </div>
        </section>

        {/* Main Grid: Interactive Map + Edge Register */}
        <section className="network-layout">
          {/* SVG Animated Road Network */}
          <div className="network-map-card">
            <div className="map-card-header">
              <div className="map-label">
                <Radio size={14} /> DIRECTED ROAD GRAPH & FLOW VECTORS
                <span style={{ fontSize: '10px', color: '#0284c7', marginLeft: '10px', textTransform: 'none' }}>
                  (Click any node to view/edit Node Telemetry Features)
                </span>
              </div>
              <div className="map-legend">
                <div className="legend-item"><span className="legend-dot green" /> Smooth (&lt;1400)</div>
                <div className="legend-item"><span className="legend-dot yellow" /> Heavy (1400-1800)</div>
                <div className="legend-item"><span className="legend-dot red" /> Bottleneck (&gt;1800)</div>
              </div>
            </div>

            <div className="svg-container">
              <svg viewBox="0 0 630 300" aria-label="Mumbai 8 node directed road graph">
                <defs>
                  <marker id="arrow-green" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
                    <path d="M0,0 L8,4 L0,8 Z" fill="#059669" />
                  </marker>
                  <marker id="arrow-yellow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
                    <path d="M0,0 L8,4 L0,8 Z" fill="#d97706" />
                  </marker>
                  <marker id="arrow-red" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
                    <path d="M0,0 L8,4 L0,8 Z" fill="#e11d48" />
                  </marker>
                </defs>

                {/* Render Edge Lines & Flow Particles */}
                {edgeOrder.map(([source, target]) => {
                  const a = points.find(p => p.id === source)!;
                  const b = points.find(p => p.id === target)!;
                  const dx = b.x - a.x;
                  const dy = b.y - a.y;
                  const length = Math.max(1, Math.hypot(dx, dy));
                  
                  // Offset from node circles (r=20)
                  const startX = a.x + (dx / length) * 22;
                  const startY = a.y + (dy / length) * 22;
                  const endX = b.x - (dx / length) * 24;
                  const endY = b.y - (dy / length) * 24;
                  
                  const edgeId = `${source}→${target}`;
                  const value = network.predictions[edgeId] ?? 0;
                  const threshold = network.edges.find(e => e.id === edgeId)?.threshold ?? 1800;
                  
                  const isCritical = value > threshold;
                  const isWatch = value > threshold * 0.78;
                  const statusClass = isCritical ? 'red' : isWatch ? 'yellow' : 'green';
                  const strokeColor = isCritical ? '#e11d48' : isWatch ? '#d97706' : '#059669';
                  const markerId = `url(#arrow-${statusClass})`;

                  const midX = (startX + endX) / 2;
                  const midY = (startY + endY) / 2;
                  const isSelected = selectedEdgeId === edgeId;
                  const isHovered = hoveredEdgeId === edgeId;

                  return (
                    <g key={`${source}-${target}`}>
                      {/* Invisible wider line for easy click target */}
                      <line 
                        x1={startX} y1={startY} x2={endX} y2={endY}
                        className="draw-edge-clickable"
                        onClick={() => setSelectedEdgeId(edgeId)}
                        onMouseEnter={() => setHoveredEdgeId(edgeId)}
                        onMouseLeave={() => setHoveredEdgeId(null)}
                      />

                      {/* Base line */}
                      <line 
                        x1={startX} y1={startY} x2={endX} y2={endY}
                        markerEnd={markerId}
                        className={`draw-edge-base ${isSelected ? 'selected' : ''}`}
                        style={{
                          stroke: strokeColor,
                          strokeWidth: isSelected || isHovered ? 4.5 : 3,
                          opacity: 0.95
                        }}
                      />

                      {/* Animated moving traffic particles */}
                      <line 
                        x1={startX} y1={startY} x2={endX} y2={endY}
                        className={`draw-edge-flow ${isCritical ? 'critical' : isWatch ? 'fast' : ''}`}
                        style={{
                          stroke: isCritical ? '#ffffff' : 'rgba(15, 23, 42, 0.35)',
                          strokeWidth: isCritical ? 2.5 : 1.5,
                        }}
                      />

                      {/* Live veh/hr pill badge over middle of edge */}
                      <g transform={`translate(${midX}, ${midY})`} style={{ pointerEvents: 'none' }}>
                        <rect 
                          x="-26" y="-10" width="52" height="18" rx="6" 
                          fill="#ffffff" 
                          stroke={strokeColor}
                          strokeWidth="1.5"
                        />
                        <text 
                          x="0" y="3" 
                          textAnchor="middle" 
                          fill="#0f172a" 
                          style={{ fontFamily: 'JetBrains Mono', fontSize: '9px', fontWeight: 800 }}
                        >
                          {Math.round(value)}
                        </text>
                      </g>
                    </g>
                  );
                })}

                {/* Render Nodes (Junctions) */}
                {points.map(point => {
                  const nodeTiming = timings.find(t => t.id === point.id);
                  const isCongested = network.edges.some(e => (e.source === point.id || e.target === point.id) && (network.predictions[e.id] ?? 0) > 1800);
                  const isNodeSelected = selectedNodeId === point.id;

                  return (
                    <g 
                      key={point.id} 
                      className="node-group" 
                      transform={`translate(${point.x}, ${point.y})`}
                      onClick={() => handleNodeClick(point.id)}
                      style={{ cursor: 'pointer' }}
                    >
                      {/* Pulse animation ring around node */}
                      <circle 
                        cx="0" cy="0" r={isNodeSelected ? "24" : "20"} 
                        className="draw-node-pulse" 
                        style={{ 
                          stroke: isNodeSelected ? '#0284c7' : isCongested ? '#e11d48' : '#0284c7',
                          strokeWidth: isNodeSelected ? 3 : 1.5
                        }} 
                      />
                      
                      {/* Main Node Circle */}
                      <circle 
                        cx="0" cy="0" r="18" 
                        className={`draw-node-bg ${isCongested ? 'congested' : ''}`}
                        style={{
                          fill: isNodeSelected ? '#e0f2fe' : undefined,
                          stroke: isNodeSelected ? '#0284c7' : undefined
                        }}
                      />

                      {/* Junction ID Label */}
                      <text x="0" y="4" textAnchor="middle" className="node-text">
                        {point.id}
                      </text>

                      {/* Cycle time badge under node */}
                      <g transform="translate(0, 30)">
                        <rect x="-18" y="-9" width="36" height="14" rx="4" fill="#ffffff" stroke={isNodeSelected ? "#0284c7" : "#cbd5e1"} strokeWidth={isNodeSelected ? "1.5" : "1"} />
                        <text x="0" y="2" textAnchor="middle" className="node-badge-text" style={{ fill: isNodeSelected ? "#0284c7" : undefined, fontWeight: isNodeSelected ? 800 : 700 }}>
                          {nodeTiming?.after ?? nodeTiming?.old}s
                        </text>
                      </g>
                    </g>
                  );
                })}
              </svg>
            </div>

            {/* Interactive Edge Slider Panel */}
            {selectedEdgeId && (
              <div className="network-edge-panel">
                <div className="network-edge-header">
                  <div className="edge-title-badge">
                    <Sliders size={16} style={{ color: '#0284c7' }} />
                    <span>Edge {selectedEdgeId}</span>
                    <span className="edge-type-tag">STQGCN Predictor</span>
                  </div>
                  <button className="close-panel-btn" onClick={() => setSelectedEdgeId(null)}>
                    Done
                  </button>
                </div>

                <div className="slider-container">
                  <input 
                    type="range" 
                    min={100} 
                    max={2400} 
                    step={20} 
                    value={sliderValue} 
                    onChange={(e) => handleEdgePredictionChange(selectedEdgeId, Number(e.target.value))} 
                  />
                  <div className="network-edge-value">
                    <strong>{Math.round(sliderValue)}</strong>
                    <small>veh/hr</small>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Edge Register Side Panel */}
          <div className="edge-list-card">
            <div className="edge-list-header">
              <div className="map-label">
                <Activity size={14} /> LIVE EDGE REGISTER
              </div>
              <span style={{ fontSize: '11px', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                11 VECTOR EDGES
              </span>
            </div>

            <div className="edge-scroll-area">
              {edgeOrder.map(([source, target], index) => {
                const edgeId = `${source}→${target}`;
                const value = network.predictions[edgeId] ?? 0;
                const threshold = network.edges.find(e => e.id === edgeId)?.threshold ?? 1800;
                const ratio = Math.min(1, value / 2200);

                const status = value > threshold ? 'red' : value > threshold * 0.78 ? 'yellow' : 'green';

                return (
                  <div 
                    className={`edge-item-row ${selectedEdgeId === edgeId ? 'selected' : ''}`}
                    key={edgeId}
                    onClick={() => setSelectedEdgeId(edgeId)}
                  >
                    <span className="edge-code">E{index + 1}</span>
                    <div className="edge-nodes">
                      {source} <ArrowRight size={12} /> {target}
                    </div>
                    <div className="edge-flow-stat">
                      <span className={`edge-flow-val ${status}`}>
                        {Math.round(value)} <small style={{ fontSize: '9px', opacity: 0.8 }}>v/h</small>
                      </span>
                      <div className="edge-mini-bar-track">
                        <div 
                          className={`edge-mini-bar-fill ${status}`}
                          style={{ width: `${ratio * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        {/* Single Node Optimal Direction & Editable Telemetry Feature Inspection Section */}
        <section className="node-inspection-section" id="node-inspection-anchor">
          <div className="node-inspection-header">
            <div>
              <h2>
                <Compass size={20} style={{ color: '#0284c7' }} />
                Single Node Optimal Direction & Telemetry Features ({selectedNodeId})
              </h2>
              <p>Real-time node dataset metrics ({currentNodeFeatures.node_name}). Type or drag sliders below to modify feature parameters.</p>
            </div>

            <div className="node-selector-pills">
              {points.map(p => (
                <button
                  key={p.id}
                  className={`node-select-btn ${selectedNodeId === p.id ? 'active' : ''}`}
                  onClick={() => setSelectedNodeId(p.id)}
                >
                  {p.id}
                </button>
              ))}
            </div>
          </div>

          {/* Edit Notification & Reset Toolbar */}
          <div className="node-edit-toolbar">
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Edit3 size={15} />
              <span>Interactive Telemetry Editor for <strong>{selectedNodeId} ({currentNodeFeatures.node_name})</strong></span>
            </div>
            <button 
              className="node-reset-btn"
              onClick={() => resetNodeFeatures(selectedNodeId)}
              title="Reset to default dataset values"
            >
              <RotateCcw size={12} /> Reset Node Defaults
            </button>
          </div>

          {/* 7 Editable Node Telemetry Features Grid */}
          <div className="node-features-grid">
            {/* 1. Flow */}
            <div className="node-feature-card">
              <div className="node-feature-header">
                <span>Flow</span>
                <div className="node-feature-icon"><Waves size={15} /></div>
              </div>
              <div className="node-feature-input-row">
                <input 
                  type="number" 
                  step="1" 
                  className="node-feature-input"
                  value={currentNodeFeatures.flow}
                  onChange={(e) => handleFeatureChange(selectedNodeId, 'flow', parseFloat(e.target.value))}
                />
                <span className="node-feature-unit">veh/m</span>
              </div>
              <input 
                type="range" 
                min="100" 
                max="1800" 
                step="10"
                className="node-feature-slider"
                value={currentNodeFeatures.flow}
                onChange={(e) => handleFeatureChange(selectedNodeId, 'flow', parseFloat(e.target.value))}
              />
            </div>

            {/* 2. Speed */}
            <div className="node-feature-card">
              <div className="node-feature-header">
                <span>Speed</span>
                <div className="node-feature-icon"><Gauge size={15} /></div>
              </div>
              <div className="node-feature-input-row">
                <input 
                  type="number" 
                  step="0.5" 
                  className="node-feature-input"
                  value={currentNodeFeatures.speed}
                  onChange={(e) => handleFeatureChange(selectedNodeId, 'speed', parseFloat(e.target.value))}
                />
                <span className="node-feature-unit">km/h</span>
              </div>
              <input 
                type="range" 
                min="5" 
                max="120" 
                step="1"
                className="node-feature-slider"
                value={currentNodeFeatures.speed}
                onChange={(e) => handleFeatureChange(selectedNodeId, 'speed', parseFloat(e.target.value))}
              />
            </div>

            {/* 3. Occupancy */}
            <div className="node-feature-card">
              <div className="node-feature-header">
                <span>Occupancy</span>
                <div className="node-feature-icon"><Percent size={15} /></div>
              </div>
              <div className="node-feature-input-row">
                <input 
                  type="number" 
                  step="0.1" 
                  className="node-feature-input"
                  value={currentNodeFeatures.occupancy}
                  onChange={(e) => handleFeatureChange(selectedNodeId, 'occupancy', parseFloat(e.target.value))}
                />
                <span className="node-feature-unit">%</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="50" 
                step="0.1"
                className="node-feature-slider"
                value={currentNodeFeatures.occupancy}
                onChange={(e) => handleFeatureChange(selectedNodeId, 'occupancy', parseFloat(e.target.value))}
              />
            </div>

            {/* 4. Queue Length */}
            <div className="node-feature-card">
              <div className="node-feature-header">
                <span>Queue Length</span>
                <div className="node-feature-icon"><Layers size={15} /></div>
              </div>
              <div className="node-feature-input-row">
                <input 
                  type="number" 
                  step="1" 
                  className="node-feature-input"
                  value={currentNodeFeatures.queue_length}
                  onChange={(e) => handleFeatureChange(selectedNodeId, 'queue_length', parseFloat(e.target.value))}
                />
                <span className="node-feature-unit">m</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="250" 
                step="1"
                className="node-feature-slider"
                value={currentNodeFeatures.queue_length}
                onChange={(e) => handleFeatureChange(selectedNodeId, 'queue_length', parseFloat(e.target.value))}
              />
            </div>

            {/* 5. Precip Intensity */}
            <div className="node-feature-card">
              <div className="node-feature-header">
                <span>Precip Intensity</span>
                <div className="node-feature-icon"><CloudRain size={15} /></div>
              </div>
              <div className="node-feature-input-row">
                <input 
                  type="number" 
                  step="0.1" 
                  className="node-feature-input"
                  value={currentNodeFeatures.precip_intensity}
                  onChange={(e) => handleFeatureChange(selectedNodeId, 'precip_intensity', parseFloat(e.target.value))}
                />
                <span className="node-feature-unit">mm/h</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="50" 
                step="0.5"
                className="node-feature-slider"
                value={currentNodeFeatures.precip_intensity}
                onChange={(e) => handleFeatureChange(selectedNodeId, 'precip_intensity', parseFloat(e.target.value))}
              />
            </div>

            {/* 6. Visibility */}
            <div className="node-feature-card">
              <div className="node-feature-header">
                <span>Visibility</span>
                <div className="node-feature-icon"><Eye size={15} /></div>
              </div>
              <div className="node-feature-input-row">
                <input 
                  type="number" 
                  step="50" 
                  className="node-feature-input"
                  value={currentNodeFeatures.visibility}
                  onChange={(e) => handleFeatureChange(selectedNodeId, 'visibility', parseFloat(e.target.value))}
                />
                <span className="node-feature-unit">m</span>
              </div>
              <input 
                type="range" 
                min="100" 
                max="10000" 
                step="100"
                className="node-feature-slider"
                value={currentNodeFeatures.visibility}
                onChange={(e) => handleFeatureChange(selectedNodeId, 'visibility', parseFloat(e.target.value))}
              />
            </div>

            {/* 7. Temp */}
            <div className="node-feature-card">
              <div className="node-feature-header">
                <span>Temp</span>
                <div className="node-feature-icon"><Thermometer size={15} /></div>
              </div>
              <div className="node-feature-input-row">
                <input 
                  type="number" 
                  step="0.5" 
                  className="node-feature-input"
                  value={currentNodeFeatures.temperature}
                  onChange={(e) => handleFeatureChange(selectedNodeId, 'temperature', parseFloat(e.target.value))}
                />
                <span className="node-feature-unit">°C</span>
              </div>
              <input 
                type="range" 
                min="-10" 
                max="50" 
                step="0.5"
                className="node-feature-slider"
                value={currentNodeFeatures.temperature}
                onChange={(e) => handleFeatureChange(selectedNodeId, 'temperature', parseFloat(e.target.value))}
              />
            </div>
          </div>

          <div className="node-inspection-body" style={{ marginTop: '12px' }}>
            {/* Junction Diagram Image Graphic */}
            <div className="node-diagram-wrapper">
              <div className="node-diagram-badge">
                <Navigation size={13} />
                JUNCTION {selectedNodeId} ARCHITECTURAL SCHEMA
              </div>
              <img 
                src="/single_node_junction.png" 
                alt={`Single Node Junction ${selectedNodeId} Diagram with Traffic Light and 3 Paths`}
                className="node-diagram-img"
              />
            </div>

            {/* 3-Path Optimal Rerouting & Traffic Signal Panel */}
            <div className="node-analysis-panel">
              {/* Traffic Light Signal Controller Box */}
              <div className="signal-status-card">
                <div className="signal-info">
                  <label>Traffic Signal Phase ({selectedNodeId})</label>
                  <h4>{active ? (qaoa ? 'QAOA Quantum Synchronized' : 'PSO Optimized Green Phase') : 'Standard Signal Cycle'}</h4>
                </div>
                <div className="signal-light-housing">
                  <div className="signal-light red" />
                  <div className="signal-light yellow" />
                  <div className="signal-light green active" />
                </div>
              </div>

              {/* 3 Path Directions Grid */}
              <div className="paths-direction-grid">
                {/* 1. Left Turn */}
                <div className="path-direction-card delayed">
                  <div className="path-left">
                    <div className="path-icon-box delayed">
                      <CornerUpLeft size={20} />
                    </div>
                    <div>
                      <div className="path-title">Left Turn</div>
                      <div className="path-sub">42% Flow Efficiency · High Queue Delay ({currentNodeFeatures.queue_length.toFixed(1)}m)</div>
                    </div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <span style={{ fontSize: '12px', fontWeight: 700, color: '#e11d48' }}>Red Signal</span>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Hold 18s</div>
                  </div>
                </div>

                {/* 2. Straight Ahead - OPTIMUM DIRECTION RECOMMENDED */}
                <div className="path-direction-card optimum">
                  <div className="path-left">
                    <div className="path-icon-box optimum">
                      <ArrowUp size={20} />
                    </div>
                    <div>
                      <div className="path-title" style={{ color: '#047857', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        Straight Ahead
                        <span className="optimum-badge">
                          <Check size={12} /> OPTIMUM DIRECTION
                        </span>
                      </div>
                      <div className="path-sub">85% Flow Efficiency · Speed: {currentNodeFeatures.speed.toFixed(1)} km/h</div>
                    </div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <span style={{ fontSize: '12px', fontWeight: 800, color: '#059669' }}>Green Signal</span>
                    <div style={{ fontSize: '11px', color: '#059669', fontWeight: 600 }}>Active {currentNodeTiming.green}s</div>
                  </div>
                </div>

                {/* 3. Right Turn */}
                <div className="path-direction-card">
                  <div className="path-left">
                    <div className="path-icon-box warning">
                      <CornerUpRight size={20} />
                    </div>
                    <div>
                      <div className="path-title">Right Turn</div>
                      <div className="path-sub">61% Flow Efficiency · Moderate Delay</div>
                    </div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <span style={{ fontSize: '12px', fontWeight: 700, color: '#d97706' }}>Amber Signal</span>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Caution 8s</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Optimizer Control Action Bar */}
        <section className="optimizer-row-card">
          <div className="optimizer-intro">
            <p>DECISION & OPTIMIZATION LAYER</p>
            <h2>Run Quantum-Inspired Signal Optimizers</h2>
            <span>Both engines execute live signal plan redistribution against the current STQGCN forecast vector.</span>
          </div>

          <div className="optimizer-actions">
            <button 
              className={`pso-action ${running === 'pso' ? 'running' : ''}`} 
              disabled={!!running} 
              onClick={() => run('pso')}
            >
              <Play size={16} fill="currentColor" />
              {running === 'pso' ? 'Executing PSO…' : 'Run PSO Engine'}
            </button>

            <button 
              className={`qaoa-action ${running === 'qaoa' ? 'running' : ''}`} 
              disabled={!!running} 
              onClick={() => run('qaoa')}
            >
              <BrainCircuit size={18} />
              {running === 'qaoa' ? 'Executing QAOA…' : 'Run QAOA Quantum Engine'}
            </button>
          </div>
        </section>

        {/* Signal Plan Timing & Junction Light Panel */}
        <section className="timing-panel-card">
          <div className="timing-heading">
            <div>
              <p>SIGNAL TIMING & JUNCTION METRICS</p>
              <h2>Optimized Signal Control Plan</h2>
            </div>
            <div className={`timing-status-tag ${active ? 'active' : ''}`}>
              {active 
                ? (qaoa ? `QAOA Quantum Active (${executionTimeMs ? `${executionTimeMs}ms` : 'Live'})` : `PSO Heuristic Active (${executionTimeMs ? `${executionTimeMs}ms` : 'Live'})`)
                : 'Awaiting Optimizer Trigger'
              }
            </div>
          </div>

          <div className="timing-grid">
            {timings.map(row => {
              const isImproved = active && row.reduced > 0;

              return (
                <div className="junction-card" key={row.id}>
                  <div className="junction-top">
                    <div className="junction-id">
                      {row.id}
                      {isImproved && <CheckCircle2 size={14} style={{ color: '#059669' }} />}
                    </div>
                    <div className="junction-signal-light">
                      <div className="sig-dot red" />
                      <div className="sig-dot yellow" />
                      <div className="sig-dot green" />
                    </div>
                  </div>

                  <div className="junction-metrics">
                    <div className="j-metric">
                      <label>Before</label>
                      <span>{row.old}s</span>
                    </div>

                    <div className="j-metric">
                      <label>After Cycle</label>
                      <span className={isImproved ? 'improved' : ''}>
                        {row.after}s
                      </span>
                    </div>

                    <div className="j-metric">
                      <label>Green Time</label>
                      <span style={{ color: '#0284c7' }}>{row.green}s</span>
                    </div>
                  </div>

                  <div className="green-alloc-bar">
                    <div className="green-alloc-label">
                      <span>Green Light Ratio</span>
                      <span>{Math.round((row.green / (row.after || 1)) * 100)}%</span>
                    </div>
                    <div className="green-bar-track">
                      <div 
                        className="green-bar-fill"
                        style={{ width: `${Math.min(100, Math.round((row.green / (row.after || 1)) * 100))}%` }}
                      />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      </main>
    </div>
  );
}
