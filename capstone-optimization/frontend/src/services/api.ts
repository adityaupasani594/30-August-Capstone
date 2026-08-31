import axios from "axios";
import { NetworkResponse, OptimizationResponse } from "../types";

const API_BASE_URL = "http://localhost:8000";

export const api = {
  getNetwork: async (networkType: string): Promise<NetworkResponse> => {
    const response = await axios.get(`${API_BASE_URL}/network`, {
      params: { network_type: networkType }
    });
    return response.data;
  },

  runOptimization: async (request: {
    capacities: Record<string, number>;
    predictions: Record<string, number>;
    network_type: string;
    force?: boolean;
  }): Promise<OptimizationResponse> => {
    const response = await axios.post(`${API_BASE_URL}/optimize`, request, {
      params: request.force ? { force: true } : undefined,
    });
    return response.data;
  },

  runQaoaOptimization: async (request: {
    capacities?: Record<string, number>;
    predictions?: Record<string, number>;
    network_type?: string;
  }): Promise<OptimizationResponse> => {
    const response = await axios.post(`${API_BASE_URL}/qaoa-optimize`, request);
    return response.data;
  },

  stqgcnPredict: async (request: {
    predictions: Record<string, number>;
    changed_edge: string;
    new_value: number;
    network_type?: string;
    edges?: Array<{ id: string; source: string; target: string; capacity: number }>;
  }): Promise<{ predictions: Record<string, number>; changed_edge: string; new_value: number }> => {
    const response = await axios.post(`${API_BASE_URL}/stqgcn-predict`, request);
    return response.data;
  },

  getResults: async (networkType: string = "vedant"): Promise<OptimizationResponse> => {
    const response = await axios.get(`${API_BASE_URL}/results`, {
      params: { network_type: networkType }
    });
    return response.data;
  },
};
