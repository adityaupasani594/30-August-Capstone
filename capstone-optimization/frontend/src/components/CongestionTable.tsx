import React from 'react';
import { EdgeData } from '../types';

interface CongestionTableProps {
  edges: EdgeData[];
  capacities: Record<string, number>;
  predicted: Record<string, number>;
  thresholds: Record<string, number>;
  optimized?: Record<string, number>;
  greenTimes?: Record<string, { old: number; new: number }>;
}

export const CongestionTable: React.FC<CongestionTableProps> = ({ edges, capacities, predicted, thresholds, optimized, greenTimes }) => {
  return (
    <div className="overflow-x-auto shadow-md sm:rounded-lg my-6">
      <table className="w-full text-sm text-left text-gray-500">
        <thead className="text-xs text-gray-700 uppercase bg-gray-50">
          <tr>
            <th scope="col" className="px-4 py-3">Edge</th>
            <th scope="col" className="px-4 py-3">Capacity</th>
            <th scope="col" className="px-4 py-3">Speed</th>
            <th scope="col" className="px-4 py-3">Lanes</th>
            <th scope="col" className="px-4 py-3">Length (km)</th>
            <th scope="col" className="px-4 py-3">Road Type</th>
            <th scope="col" className="px-4 py-3">Predicted</th>
            <th scope="col" className="px-4 py-3">Threshold</th>
            <th scope="col" className="px-4 py-3">Optimized</th>
            <th scope="col" className="px-4 py-3">Green Time (Old/New)</th>
            <th scope="col" className="px-4 py-3">Status Before</th>
            <th scope="col" className="px-4 py-3">Status After</th>
          </tr>
        </thead>
        <tbody>
          {edges.map((edge) => {
            const pred = predicted[edge.id] || 0;
            const thresh = thresholds[edge.id] || 0;
            const opt = optimized ? optimized[edge.id] : undefined;
            const currentCapacity = capacities[edge.id] ?? edge.capacity;
            const gt = greenTimes ? greenTimes[edge.id] : undefined;

            let statusBefore = edge.is_reference ? 'Reference' : pred > thresh ? 'Congested' : 'Normal';
            let statusAfter = edge.is_reference ? 'Reference' : (opt !== undefined && opt > thresh) ? 'Congested' : (opt !== undefined ? 'Normal' : '-');

            return (
              <tr key={edge.id} className="bg-white border-b hover:bg-gray-50">
                <td className="px-4 py-2 font-medium text-gray-900">{edge.id}</td>
                <td className="px-4 py-2">{currentCapacity}</td>
                <td className="px-4 py-2">{edge.speed}</td>
                <td className="px-4 py-2">{edge.lanes}</td>
                <td className="px-4 py-2">{edge.length}</td>
                <td className="px-4 py-2">{edge.road_type}</td>
                <td className="px-4 py-2">{Math.round(pred)}</td>
                <td className="px-4 py-2">{Math.round(thresh)}</td>
                <td className="px-4 py-2">{opt !== undefined ? Math.round(opt) : '-'}</td>
                <td className="px-4 py-2 font-medium text-gray-800">{gt ? `${gt.old}s / ${gt.new}s` : '-'}</td>
                <td className={`px-4 py-2 font-semibold ${statusBefore === 'Congested' ? 'text-red-600' : statusBefore === 'Normal' ? 'text-green-600' : 'text-gray-500'}`}>
                  {statusBefore}
                </td>
                <td className={`px-4 py-2 font-semibold ${statusAfter === 'Congested' ? 'text-red-600' : statusAfter === 'Normal' ? 'text-green-600' : 'text-gray-500'}`}>
                  {statusAfter}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};
