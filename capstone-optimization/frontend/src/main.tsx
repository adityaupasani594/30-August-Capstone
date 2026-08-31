/**
 * main.tsx
 * ========
 * React application entry point.
 * Mounts <App /> into the #root div defined in index.html.
 */

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';
import App from './App.tsx';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
