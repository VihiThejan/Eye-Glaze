import express from 'express';
import { 
  getRecommendations, 
  getEyeHealthSummary,
  getUserRecommendationHistory
} from '../controllers/recommendations.controller.js';

const router = express.Router();

// Recommendations endpoints
router.get('/user/:username', getRecommendations);
router.get('/summary/:username', getEyeHealthSummary);
router.get('/history/:username', getUserRecommendationHistory);

export default router;