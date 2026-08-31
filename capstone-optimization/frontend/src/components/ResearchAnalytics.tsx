import React from 'react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LabelList
} from 'recharts';
import { ScenarioResult } from '../types';

interface ResearchAnalyticsProps {
  scenarios: ScenarioResult[];
  scenarioName: string;
  onChangeScenarioName: (name: string) => void;
  onClearHistory: () => void;
}

export const ResearchAnalytics: React.FC<ResearchAnalyticsProps> = ({
  scenarios,
  scenarioName,
  onChangeScenarioName,
  onClearHistory
}) => {

  const getConvergenceIteration = (history: number[]) => {
    if (!history || history.length === 0) return 0;
    const minFitness = Math.min(...history);
    const idx = history.findIndex(f => Math.abs(f - minFitness) < 1e-5);
    return idx >= 0 ? idx : 0;
  };

  // Multi-scenario convergence line chart data
  const maxIterLength = Math.max(...scenarios.map(s => s.fitnessHistory?.length || 0), 0);
  const multiScenarioLineData = Array.from({ length: maxIterLength }, (_, iter) => {
    const row: Record<string, any> = { iteration: iter };
    let sumFit = 0;
    let count = 0;
    scenarios.forEach(s => {
      if (s.fitnessHistory && iter < s.fitnessHistory.length) {
        row[s.scenarioName] = s.fitnessHistory[iter];
        sumFit += s.fitnessHistory[iter];
        count++;
      }
    });
    row['Mean'] = count > 0 ? sumFit / count : 0;
    return row;
  });

  const scenarioColors = [
    '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316'
  ];

  // Utility to trigger CSV download
  const downloadCSV = (filename: string, content: string) => {
    const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", filename);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Export summary table CSV
  const handleExportSummary = () => {
    let csvContent = "Scenario,Initial Fitness,Final Fitness,Fitness Improvement (%),Congested Edge Count\n";
    scenarios.forEach(s => {
      csvContent += `"${s.scenarioName}",${s.initialFitness},${s.finalFitness},${s.fitnessImprovement.toFixed(2)},${s.congestedEdges}\n`;
    });
    downloadCSV("scenario_comparison_summary.csv", csvContent);
  };

  // Export a single scenario's fitness history
  const handleExportSingleHistory = (scenario: ScenarioResult) => {
    let csvContent = "Iteration,Fitness\n";
    scenario.fitnessHistory.forEach((fit, iter) => {
      csvContent += `${iter},${fit}\n`;
    });
    const safeName = scenario.scenarioName.replace(/[^a-z0-9]/gi, '_').toLowerCase();
    downloadCSV(`${safeName}_fitness_history.csv`, csvContent);
  };

  // Global Export Results: downloads summary CSV and separate CSVs for each scenario's history
  const handleExportAll = () => {
    handleExportSummary();
    scenarios.forEach((s, idx) => {
      setTimeout(() => {
        handleExportSingleHistory(s);
      }, (idx + 1) * 300);
    });
  };

  if (scenarios.length === 0) {
    return (
      <div className="bg-white p-6 rounded-lg shadow border border-gray-200 mt-8 mb-8">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center border-b border-gray-200 pb-4 mb-6 gap-4">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 text-left">Research Analytics</h2>
            <p className="text-sm text-gray-500 mt-1 text-left font-medium">Comparative statistics and optimization metrics across multiple runs</p>
          </div>
        </div>

        <div className="mb-6 bg-gray-50 p-4 rounded-lg border border-gray-200 flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-inner">
          <div className="flex items-center space-x-3 w-full md:w-auto">
            <label htmlFor="scenario-name-input" className="text-gray-700 font-semibold shrink-0">Scenario Name:</label>
            <input
              id="scenario-name-input"
              type="text"
              value={scenarioName}
              onChange={(e) => onChangeScenarioName(e.target.value)}
              placeholder="Type scenario name (e.g. Morning Peak)..."
              className="border border-gray-300 rounded px-3 py-2 bg-white text-gray-700 font-medium focus:outline-none focus:ring-2 focus:ring-blue-500 w-72 shadow-sm"
            />
          </div>
          <div className="text-sm text-gray-500 italic text-left md:text-right">
            Label optimization runs to compare scenarios dynamically on the charts.
          </div>
        </div>

        <div className="p-8 text-center bg-gray-50 rounded-lg border border-dashed border-gray-300">
          <p className="text-gray-500 italic">No experimental data collected yet. Enter a scenario name above and click "Run Optimization" to populate the research analytics dashboard.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow border border-gray-200 mt-8 mb-8">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center border-b border-gray-200 pb-4 mb-6 gap-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 text-left">Research Analytics</h2>
          <p className="text-sm text-gray-500 mt-1 text-left font-medium">Comparative statistics and optimization metrics across multiple runs</p>
        </div>
        <div className="flex items-center space-x-3 w-full md:w-auto self-end md:self-auto">
          <button
            onClick={onClearHistory}
            className="bg-red-50 hover:bg-red-100 text-red-600 hover:text-red-700 font-bold py-2 px-4 rounded border border-red-200 shadow-sm flex items-center text-sm transition-colors cursor-pointer"
          >
            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
            </svg>
            Clear History
          </button>
          <button
            onClick={handleExportAll}
            className="bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-2 px-5 rounded shadow flex items-center text-sm transition-colors cursor-pointer"
          >
            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
            </svg>
            Export Results
          </button>
        </div>
      </div>

      <div className="mb-6 bg-gray-50 p-4 rounded-lg border border-gray-200 flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-inner">
        <div className="flex items-center space-x-3 w-full md:w-auto">
          <label htmlFor="scenario-name-input-active" className="text-gray-700 font-semibold shrink-0">Scenario Name:</label>
          <input
            id="scenario-name-input-active"
            type="text"
            value={scenarioName}
            onChange={(e) => onChangeScenarioName(e.target.value)}
            placeholder="Type scenario name (e.g. Morning Peak)..."
            className="border border-gray-300 rounded px-3 py-2 bg-white text-gray-700 font-medium focus:outline-none focus:ring-2 focus:ring-blue-500 w-72 shadow-sm"
          />
        </div>
        <div className="text-sm text-gray-500 italic text-left md:text-right font-medium">
          Label optimization runs to compare scenarios dynamically on the charts.
        </div>
      </div>

      {/* Grid of cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        
        {/* 1. Multi-Scenario Convergence Line Chart */}
        <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 h-[360px]">
          <h3 className="text-lg font-bold text-gray-800 mb-2 text-left">PSO Multi-Scenario Convergence Plot</h3>
          <div className="h-[280px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={multiScenarioLineData} margin={{ top: 15, right: 30, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="iteration" label={{ value: 'Iteration', position: 'bottom', offset: 5, style: { fontSize: '12px', fill: '#6b7280', fontWeight: 600 } }} />
                <YAxis label={{ value: 'Fitness (Peak % + Penalty)', angle: -90, position: 'left', offset: 10, style: { fontSize: '12px', fill: '#6b7280', fontWeight: 600 } }} />
                <Tooltip formatter={(value: any) => [Number(value).toFixed(2), "Fitness"]} />
                <Legend verticalAlign="top" height={36} />
                {scenarios.map((s, idx) => (
                  <Line
                    key={s.scenarioName}
                    type="monotone"
                    dataKey={s.scenarioName}
                    stroke={scenarioColors[idx % scenarioColors.length]}
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    name={s.scenarioName}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 2. PSO Convergence Iterations Table */}
        <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 flex flex-col h-[360px]">
          <h3 className="text-lg font-bold text-gray-800 mb-3 text-left">PSO Convergence Iterations</h3>
          <div className="overflow-y-auto border border-gray-200 rounded-lg flex-grow bg-white shadow-inner">
            <table className="w-full text-sm text-left text-gray-500">
              <thead className="text-xs text-gray-700 uppercase bg-gray-50 sticky top-0 border-b border-gray-200">
                <tr>
                  <th scope="col" className="px-4 py-3">Scenario</th>
                  <th scope="col" className="px-4 py-3 text-center">Convergence Iteration</th>
                </tr>
              </thead>
              <tbody>
                {scenarios.map((s, idx) => {
                  const convergenceIter = getConvergenceIteration(s.fitnessHistory);
                  return (
                    <tr key={s.scenarioName + idx} className="border-b hover:bg-gray-50 bg-white">
                      <td className="px-4 py-3 font-semibold text-gray-900">{s.scenarioName}</td>
                      <td className="px-4 py-3 text-center font-bold text-blue-600 text-base">{convergenceIter}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* 3. Initial vs Final Fitness */}
        <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 h-[360px]">
          <h3 className="text-lg font-bold text-gray-800 mb-4 text-left">Initial vs Final Fitness</h3>
          <div className="h-[280px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={scenarios} margin={{ top: 15, right: 30, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="scenarioName" label={{ value: 'Scenario', position: 'bottom', offset: 5, style: { fontSize: '12px', fill: '#6b7280', fontWeight: 600 } }} />
                <YAxis label={{ value: 'Fitness Value', angle: -90, position: 'left', offset: 0, style: { fontSize: '12px', fill: '#6b7280', fontWeight: 600 } }} />
                <Tooltip />
                <Legend verticalAlign="top" height={36}/>
                <Bar dataKey="initialFitness" fill="#94a3b8" name="Initial Fitness" />
                <Bar dataKey="finalFitness" fill="#3b82f6" name="Final Fitness" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 4. Percentage Fitness Improvement */}
        <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 h-[360px]">
          <h3 className="text-lg font-bold text-gray-800 mb-4 text-left">Percentage Fitness Improvement</h3>
          <div className="h-[280px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={scenarios} margin={{ top: 25, right: 30, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="scenarioName" label={{ value: 'Scenario', position: 'bottom', offset: 5, style: { fontSize: '12px', fill: '#6b7280', fontWeight: 600 } }} />
                <YAxis label={{ value: 'Fitness Improvement (%)', angle: -90, position: 'left', offset: 0, style: { fontSize: '12px', fill: '#6b7280', fontWeight: 600 } }} unit="%" />
                <Tooltip formatter={(value: any) => [`${parseFloat(value).toFixed(2)}%`, "Improvement"]} />
                <Bar dataKey="fitnessImprovement" fill="#10b981" name="Fitness Improvement (%)">
                  <LabelList
                    dataKey="fitnessImprovement"
                    position="top"
                    formatter={(val: any) => val !== undefined && val !== null ? `${Number(val).toFixed(1)}%` : ''}
                    style={{ fontSize: '11px', fontWeight: 'bold', fill: '#065f46' }}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Scenario Table */}
      <div className="mt-8 border-t border-gray-200 pt-6">
        <h3 className="text-lg font-bold text-gray-800 mb-4 text-left">Scenario Comparison Table</h3>
        <div className="overflow-x-auto border border-gray-200 rounded-lg">
          <table className="w-full text-sm text-left text-gray-500">
            <thead className="text-xs text-gray-700 uppercase bg-gray-50">
              <tr>
                <th scope="col" className="px-4 py-3">Scenario</th>
                <th scope="col" className="px-4 py-3 text-center">Initial Fitness</th>
                <th scope="col" className="px-4 py-3 text-center">Final Fitness</th>
                <th scope="col" className="px-4 py-3 text-center">Fitness Improvement</th>
                <th scope="col" className="px-4 py-3 text-center">Congested Edge Count</th>
                <th scope="col" className="px-4 py-3 text-center">Export Data</th>
              </tr>
            </thead>
            <tbody>
              {scenarios.map((s, idx) => (
                <tr key={s.scenarioName + idx} className="bg-white border-b hover:bg-gray-50">
                  <td className="px-4 py-3 font-semibold text-gray-900">{s.scenarioName}</td>
                  <td className="px-4 py-3 text-center font-medium text-gray-700">{s.initialFitness.toFixed(2)}</td>
                  <td className="px-4 py-3 text-center font-medium text-gray-700">{s.finalFitness.toFixed(2)}</td>
                  <td className="px-4 py-3 text-center font-semibold text-emerald-600">{s.fitnessImprovement.toFixed(2)}%</td>
                  <td className="px-4 py-3 text-center font-medium text-gray-700">{s.congestedEdges}</td>
                  <td className="px-4 py-3 text-center">
                    <button
                      onClick={() => handleExportSingleHistory(s)}
                      className="bg-blue-50 hover:bg-blue-100 text-blue-600 hover:text-blue-700 font-semibold py-1.5 px-3 rounded border border-blue-200 text-xs transition-colors inline-flex items-center cursor-pointer"
                    >
                      <svg className="w-3.5 h-3.5 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                      </svg>
                      History CSV
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

