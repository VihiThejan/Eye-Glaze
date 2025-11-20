import express from 'express';
import dotenv from 'dotenv';
import cors from 'cors';
import authRoutes from './routes/auth.routes.js'
import analysisRoutes from './routes/analysis.routes.js'
import uploadRoutes from './routes/upload.routes.js'
import recommendationsRoutes from './routes/recommendations.routes.js'
import { connectToDatabase, client } from './configs/db.config.js';

dotenv.config();
const app = express();
const port = process.env.EXPRESS_PORT || 5176;

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Health check route
app.get('/', (req, res) => {
  res.status(200).json({ status: 'API is running' });
});

// Detailed health and status route
app.get('/api/health', async (req, res) => {
  try {
    // Check MongoDB connection
    const isConnected = client.topology && client.topology.isConnected();
    
    // Get system information
    const healthInfo = {
      status: 'operational',
      timestamp: new Date(),
      serverInfo: {
        uptime: process.uptime(),
        memoryUsage: process.memoryUsage(),
        nodeVersion: process.version,
      },
      database: {
        connected: isConnected,
        name: isConnected ? 'eye-glaze-db' : null
      }
    };
    
    res.status(200).json(healthInfo);
  } catch (error) {
    console.error('Health check error:', error);
    res.status(500).json({
      status: 'error',
      message: 'Error getting system health',
      error: error.message
    });
  }
});

// Routes Handlers
app.use('/api/auth', authRoutes);
app.use('/api/analysis', analysisRoutes);
app.use('/api/upload', uploadRoutes);
app.use('/api/recommendations', recommendationsRoutes);

// Global error handler
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    status: 'error',
    message: err.message || 'Internal Server Error'
  });
});

// MongoDB connection status route
app.get('/api/status/db', (req, res) => {
  try {
    const db = client?.topology?.s?.state;
    const isConnected = db === 'connected';
    
    res.status(200).json({
      status: isConnected ? 'success' : 'error',
      message: isConnected ? 'Connected to MongoDB' : 'Not connected to MongoDB',
      connectionState: db || 'unknown'
    });
  } catch (err) {
    res.status(500).json({
      status: 'error',
      message: 'Error checking MongoDB connection',
      error: err.message
    });
  }
});

// Start server
connectToDatabase()
  .then(() => {
    app.listen(port, () => {
      console.log(`Server running on port: http://localhost:${port}`);
      console.log(`MongoDB Connection Status: http://localhost:${port}/api/status/db`);
    });
  })
  .catch(err => {
    console.error('Failed to connect to the database:', err);
    process.exit(1);
  });
