import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { OptimizationResponse } from '../types';

interface OptimizationSummaryProps {
  result: OptimizationResponse;
}

export const OptimizationSummary: React.FC<OptimizationSummaryProps> = ({ result }) => {
  const {
    optimization_triggered,
    iterations,
    initial_fitness,
    final_fitness,
    fitness_history,
    cycle_times,
    green_times,
  } = result;

  if (!optimization_triggered) {
    return (
      <div className="p-4 mb-4 text-sm text-green-800 rounded-lg bg-green-50">
        <span className="font-medium">Optimization Status:</span> No Optimization Required. All edges are operating below their capacity thresholds.
      </div>
    );
  }

  const fitnessImprovement = initial_fitness && final_fitness 
    ? ((initial_fitness - final_fitness) / initial_fitness * 100).toFixed(2)
    : '0.00';

  const convergenceData = fitness_history?.map((fit, iter) => ({
    iteration: iter,
    fitness: fit,
  })) || [];

  return (
    <div className="bg-white p-6 shadow-md sm:rounded-lg mb-6 border border-gray-200">
      <h3 className="text-xl font-bold mb-4 text-gray-800">Optimization Summary</h3>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-gray-50 p-4 rounded text-center border border-gray-100">
          <div className="text-sm text-gray-500 uppercase">Triggered</div>
          <div className="text-lg font-semibold text-blue-600">Yes</div>
        </div>
        <div className="bg-gray-50 p-4 rounded text-center border border-gray-100">
          <div className="text-sm text-gray-500 uppercase">Iterations</div>
          <div className="text-lg font-semibold text-gray-900">{iterations}</div>
        </div>
        <div className="bg-gray-50 p-4 rounded text-center border border-gray-100">
          <div className="text-sm text-gray-500 uppercase">Initial Fitness</div>
          <div className="text-lg font-semibold text-red-500">{initial_fitness}</div>
        </div>
        <div className="bg-gray-50 p-4 rounded text-center border border-gray-100">
          <div className="text-sm text-gray-500 uppercase">Final Fitness</div>
          <div className="text-lg font-semibold text-green-600">{final_fitness}</div>
        </div>
      </div>

      <div className="mb-6">
        <span className="font-semibold text-gray-700">Overall Fitness Improvement: </span>
        <span className="text-lg text-green-600 font-bold">{fitnessImprovement}%</span>
      </div>

      {/* PSO Fitness Convergence Plot */}
      {convergenceData.length > 0 && (
        <div className="mb-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
          <h4 className="text-lg font-bold text-gray-800 mb-3 text-left">PSO Swarm Fitness Convergence Curve</h4>
          <div className="h-[280px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={convergenceData} margin={{ top: 15, right: 30, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis 
                  dataKey="iteration" 
                  label={{ value: 'Iteration', position: 'bottom', offset: 5, style: { fontSize: '12px', fill: '#6b7280', fontWeight: 600 } }} 
                />
                <YAxis 
                  label={{ value: 'Fitness (Peak % + Penalty)', angle: -90, position: 'left', offset: 10, style: { fontSize: '12px', fill: '#6b7280', fontWeight: 600 } }} 
                />
                <Tooltip formatter={(value: any) => [Number(value).toFixed(2), "Fitness"]} />
                <Line 
                  type="monotone" 
                  dataKey="fitness" 
                  stroke="#2563eb" 
                  strokeWidth={3} 
                  dot={{ r: 4, fill: '#2563eb' }} 
                  activeDot={{ r: 7 }} 
                  name="PSO Fitness" 
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <h4 className="text-lg font-semibold mb-3 text-gray-800">Cycle Times (seconds)</h4>
      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left text-gray-500">
          <thead className="text-xs text-gray-700 uppercase bg-gray-50">
            <tr>
              <th scope="col" className="px-4 py-2">Node</th>
              <th scope="col" className="px-4 py-2">Old</th>
              <th scope="col" className="px-4 py-2">New</th>
              <th scope="col" className="px-4 py-2">% Improvement</th>
            </tr>
          </thead>
          <tbody>
            {cycle_times && Object.entries(cycle_times).map(([node, data]) => {
              const diff = data.old - data.new;
              const pctChange = ((data.new - data.old) / data.old * 100).toFixed(1);
              const colorClass = diff > 0 ? 'text-green-600' : diff < 0 ? 'text-red-600' : 'text-gray-500';

              return (
                <tr key={node} className="bg-white border-b">
                  <td className="px-4 py-2 font-medium text-gray-900">{node}</td>
                  <td className="px-4 py-2">{data.old}</td>
                  <td className="px-4 py-2 font-bold">{data.new}</td>
                  <td className={`px-4 py-2 font-semibold ${colorClass}`}>
                    {pctChange}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {green_times && Object.keys(green_times).length > 0 && (
        <>
          <h4 className="text-lg font-semibold mt-6 mb-3 text-gray-800">Dynamic Green Times (seconds)</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left text-gray-500">
              <thead className="text-xs text-gray-700 uppercase bg-gray-50">
                <tr>
                  <th scope="col" className="px-4 py-2">Edge (Approach)</th>
                  <th scope="col" className="px-4 py-2">Old Green Time</th>
                  <th scope="col" className="px-4 py-2">New Green Time</th>
                  <th scope="col" className="px-4 py-2">% Change</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(green_times).map(([edge, data]) => {
                  const diff = data.new - data.old;
                  const pctChange = data.old !== 0 ? ((data.new - data.old) / data.old * 100).toFixed(1) : '0.0';
                  const colorClass = diff > 0 ? 'text-green-600' : diff < 0 ? 'text-red-600' : 'text-gray-500';

                  return (
                    <tr key={edge} className="bg-white border-b">
                      <td className="px-4 py-2 font-medium text-gray-900">{edge}</td>
                      <td className="px-4 py-2">{data.old}s</td>
                      <td className="px-4 py-2 font-bold">{data.new}s</td>
                      <td className={`px-4 py-2 font-semibold ${colorClass}`}>
                        {diff > 0 ? '+' : ''}{pctChange}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
};

