import React from 'react';
import { EdgeData } from '../types';

interface CapacityConfigProps {
  edges: EdgeData[];
  capacities: Record<string, number>;
  predictions: Record<string, number>;
  onCapacityChange: (edgeId: string, newCapacity: number) => void;
  onPredictionChange: (edgeId: string, newPrediction: number) => void;
}

export const CapacityConfig: React.FC<CapacityConfigProps> = ({ 
  edges, 
  capacities, 
  predictions,
  onCapacityChange,
  onPredictionChange 
}) => {
  
  const handleCapacityInputChange = (edgeId: string, value: string) => {
    const newCapacity = parseFloat(value) || 0;
    onCapacityChange(edgeId, newCapacity);
  };

  const handlePredictionInputChange = (edgeId: string, value: string) => {
    const newPrediction = parseFloat(value) || 0;
    onPredictionChange(edgeId, newPrediction);
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow border border-gray-200 mb-6">
      <h2 className="text-xl font-bold mb-4 text-gray-800">Network Configuration</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left text-gray-500">
          <thead className="text-xs text-gray-700 uppercase bg-gray-50">
            <tr>
              <th scope="col" className="px-4 py-3 w-1/4">Edge</th>
              <th scope="col" className="px-4 py-3 w-3/8">Road Capacity (veh/hr)</th>
              <th scope="col" className="px-4 py-3 w-3/8">Predicted Traffic (veh/hr)</th>
            </tr>
          </thead>
          <tbody>
            {edges.map((edge) => (
              <tr key={edge.id} className="bg-white border-b hover:bg-gray-50">
                <td className="px-4 py-2 font-medium text-gray-900">{edge.id}</td>
                <td className="px-4 py-2">
                  <input
                    type="number"
                    min="1"
                    step="100"
                    className="border border-gray-300 rounded px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500 w-48 text-gray-700 font-semibold"
                    value={capacities[edge.id] ?? edge.capacity}
                    onChange={(e) => handleCapacityInputChange(edge.id, e.target.value)}
                  />
                </td>
                <td className="px-4 py-2">
                  <input
                    type="number"
                    min="0"
                    step="100"
                    className="border border-gray-300 rounded px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500 w-48 text-gray-700 font-semibold"
                    value={predictions[edge.id] ?? 0}
                    onChange={(e) => handlePredictionInputChange(edge.id, e.target.value)}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
