import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

/**
 * vite.config.ts
 * ==============
 * Vite configuration for the PSO Traffic Optimization frontend.
 *
 * Plugins:
 *   react        — JSX transform, Fast Refresh HMR
 *   tailwindcss  — Tailwind CSS v4 native Vite integration
 *
 * Dev server:
 *   Runs on http://localhost:5173 (Vite default).
 *   The FastAPI backend runs on http://localhost:8000.
 */
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    port: 5173,
  },
});
