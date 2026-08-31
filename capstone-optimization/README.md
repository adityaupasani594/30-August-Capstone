# Traffic Signal Optimization

Research prototype for a novel Particle Swarm Optimization (PSO)-based traffic signal optimization framework. 

This project simulates traffic redistribution using a deterministic model and optimizes traffic signal cycle times on a 5-node demonstration network to minimize overall congestion variance.

## Project Structure

- `backend/`: FastAPI Python server containing the core PSO algorithm, threshold logic, and mathematical formulations.
- `frontend/`: Minimalist React & Vite application visualizing the network graph and optimization results.

## Quick Start

To run the application, you need to start both the backend server and the frontend development server in two separate terminal windows.

### 1. Start the Backend

Open your first terminal and run the following commands:

```bash
cd backend
pip install -r requirements.txt  # Install dependencies if you haven't already
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
The backend API will now be running on `http://localhost:8000`.

### 2. Start the Frontend

Open a second terminal window and run:

```bash
cd frontend
npm install  # Install dependencies if you haven't already
npm run dev
```
The frontend dashboard will now be running (usually on `http://localhost:5173`). Open the provided link in your browser to interact with the prototype.
