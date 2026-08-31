import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '../capstone-optimization/frontend/src/style.css';
import App from './App';

createRoot(document.getElementById('root')!).render(
  <StrictMode><App /></StrictMode>,
);
