import React from 'react';
import { EdgeData } from '../types';

interface ResearchMetricsTableProps {
  edges: EdgeData[];
  predictions: Record<string, number>;
  thresholds: Record<string, number>;
  result: any | null;
}

export const ResearchMetricsTable: React.FC<ResearchMetricsTableProps> = ({
  edges,
  predictions,
  thresholds,
  result,
}) => {
  const numEdges = edges.length;

  // 1. Number of congested edges
  const congestedBefore = edges.filter(e => predictions[e.id] > (thresholds[e.id] || 0)).length;
  const congestedAfter = result && result.optimization_triggered
    ? edges.filter(e => result.after[e.id] > (result.thresholds?.[e.id] || thresholds[e.id] || 0)).length
    : null;

  // 2. Average congestion (veh/hr)
  const avgCongestionBefore = numEdges > 0
    ? Object.values(predictions).reduce((a, b) => a + b, 0) / numEdges
    : 0;
  const avgCongestionAfter = result && result.optimization_triggered
    ? (Object.values(result.after) as number[]).reduce((a, b) => a + b, 0) / numEdges
    : null;

  // 3. Maximum congestion (veh/hr)
  const maxCongestionBefore = numEdges > 0
    ? Math.max(...Object.values(predictions))
    : 0;
  const maxCongestionAfter = result && result.optimization_triggered
    ? Math.max(...Object.values(result.after).map(Number))
    : null;

  // 4. Total network congestion
  const totalCongestionBefore = Object.values(predictions).reduce((a, b) => a + b, 0);
  const totalCongestionAfter = result && result.optimization_triggered
    ? (Object.values(result.after) as number[]).reduce((a, b) => a + b, 0)
    : null;

  // 5. Mean congestion-to-threshold ratio
  const meanRatioBefore = numEdges > 0
    ? edges.reduce((acc, e) => {
        const threshold = thresholds[e.id] || 1;
        return acc + (predictions[e.id] / threshold);
      }, 0) / numEdges
    : 0;
  const meanRatioAfter = result && result.optimization_triggered
    ? edges.reduce((acc, e) => {
        const threshold = result.thresholds?.[e.id] || thresholds[e.id] || 1;
        return acc + ((result.after[e.id] || 0) / threshold);
      }, 0) / numEdges
    : null;

  // 6. Average signal cycle time (s)
  const avgCycleTimeAfter = result && result.optimization_triggered && result.cycle_times
    ? Object.values(result.cycle_times).reduce((acc: number, item: any) => acc + item.new, 0) / Object.keys(result.cycle_times).length
    : null;

  // 7. Fitness value
  const fitnessBefore = result ? result.initial_fitness : null;
  const fitnessAfter = result ? result.final_fitness : null;

  // 8. PSO iterations to convergence
  const iterationsAfter = result ? result.iterations : null;

  // 9. Reduction in peak (%)
  const reductionPeak = result && result.optimization_triggered && maxCongestionBefore > 0 && maxCongestionAfter !== null
    ? ((maxCongestionBefore - maxCongestionAfter) / maxCongestionBefore) * 100
    : null;

  // 10. Improvement of load distribution (%)
  const improvementLoadDistribution = result && result.optimization_triggered && fitnessBefore && fitnessAfter && fitnessBefore > 0
    ? ((fitnessBefore - fitnessAfter) / fitnessBefore) * 100
    : null;

  const formatVal = (val: number | null, decimals = 2, isPercentage = false) => {
    if (val === null || val === undefined) return '—';
    return val.toFixed(decimals) + (isPercentage ? '%' : '');
  };

  const formatInt = (val: number | null) => {
    if (val === null || val === undefined) return '—';
    return val.toString();
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow border border-gray-200 mt-6">
      <h2 className="text-xl font-bold mb-4 text-gray-800">Research Evaluation Metrics</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left text-gray-500">
          <thead className="text-xs text-gray-700 uppercase bg-gray-50">
            <tr>
              <th scope="col" className="px-4 py-3 w-1/2">Metric</th>
              <th scope="col" className="px-4 py-3 text-center w-1/4">Before PSO</th>
              <th scope="col" className="px-4 py-3 text-center w-1/4">After PSO</th>
            </tr>
          </thead>
          <tbody>
            <tr className="bg-white border-b hover:bg-gray-50">
              <td className="px-4 py-3 font-medium text-gray-900">Number of congested edges</td>
              <td className="px-4 py-3 text-center font-semibold text-gray-700">{formatInt(congestedBefore)}</td>
              <td className="px-4 py-3 text-center font-semibold text-gray-700">{formatInt(congestedAfter)}</td>
            </tr>
            <tr className="bg-white border-b hover:bg-gray-50">
              <td className="px-4 py-3 font-medium text-gray-900">Average congestion (veh/hr)</td>
              <td className="px-4 py-3 text-center font-semibold text-gray-700">{formatVal(avgCongestionBefore)}</td>
              <td className="px-4 py-3 text-center font-semibold text-gray-700">{formatVal(avgCongestionAfter)}</td>
            </tr>
            <tr className="bg-white border-b hover:bg-gray-50">
              <td className="px-4 py-3 font-medium text-gray-900">Maximum congestion (veh/hr)</td>
              <td className="px-4 py-3 text-center font-semibold text-gray-700">{formatVal(maxCongestionBefore)}</td>
              <td className="px-4 py-3 text-center font-semibold text-gray-700">{formatVal(maxCongestionAfter)}</td>
            </tr>
            <tr className="bg-white border-b hover:bg-gray-50">
              <td className="px-4 py-3 font-medium text-gray-900">Total network congestion</td>
              <td className="px-4 py-3 text-center font-semibold text-gray-700">{formatVal(totalCongestionBefore)}</td>
              <td className="px-4 py-3 text-center font-semibold text-gray-700">{formatVal(totalCongestionAfter)}</td>
            </tr>
            <tr className="bg-white border-b hover:bg-gray-50">
              <td className="px-4 py-3 font-medium text-gray-900">Mean congestion-to-threshold ratio</td>
              <td className="px-4 py-3 text-center font-semibold text-gray-700">{formatVal(meanRatioBefore)}</td>
              <td className="px-4 py-3 text-center font-semibold text-gray-700">{formatVal(meanRatioAfter)}</td>
            </tr>
            <tr className="bg-white border-b hover:bg-gray-50">
              <td className="px-4 py-3 font-medium text-gray-900">Average signal cycle time (s)</td>
              <td className="px-4 py-3 text-center font-semibold text-gray-700">—</td>
              <td className="px-4 py-3 text-center font-semibold text-gray-700">{formatVal(avgCycleTimeAfter)}</td>
            </tr>
            <tr className="bg-white border-b hover:bg-gray-50">
              <td className="px-4 py-3 font-medium text-gray-900">Fitness value</td>
              <td className="px-4 py-3 text-center font-semibold text-gray-700">{formatVal(fitnessBefore)}</td>
              <td className="px-4 py-3 text-center font-semibold text-gray-700">{formatVal(fitnessAfter)}</td>
            </tr>
            <tr className="bg-white border-b hover:bg-gray-50">
              <td className="px-4 py-3 font-medium text-gray-900">PSO iterations to convergence</td>
              <td className="px-4 py-3 text-center font-semibold text-gray-700">—</td>
              <td className="px-4 py-3 text-center font-semibold text-gray-700">{formatInt(iterationsAfter)}</td>
            </tr>
            <tr className="bg-white border-b hover:bg-gray-50">
              <td className="px-4 py-3 font-medium text-gray-900">Reduction in peak</td>
              <td className="px-4 py-3 text-center font-semibold text-gray-700">—</td>
              <td className="px-4 py-3 text-center font-semibold text-gray-700">{formatVal(reductionPeak, 2, true)}</td>
            </tr>
            <tr className="bg-white border-b hover:bg-gray-50">
              <td className="px-4 py-3 font-medium text-gray-900">Improvement of load distribution</td>
              <td className="px-4 py-3 text-center font-semibold text-gray-700">—</td>
              <td className="px-4 py-3 text-center font-semibold text-gray-700">{formatVal(improvementLoadDistribution, 2, true)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};
