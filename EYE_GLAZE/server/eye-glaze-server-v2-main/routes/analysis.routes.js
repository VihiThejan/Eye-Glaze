import express from 'express';
import { submitAnalysis, getUserAnalyses, getLatestAnalysis, getUserAnalysisCount } from '../controllers/analysis.controller.js';

const router = express.Router();

// Analysis endpoints
router.post('/submit', submitAnalysis);
router.get('/user/:username', getUserAnalyses);
router.get('/latest/:username', getLatestAnalysis);
router.get('/count/:username', getUserAnalysisCount);

export default router;