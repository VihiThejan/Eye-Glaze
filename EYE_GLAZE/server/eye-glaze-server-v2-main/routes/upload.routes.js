import express from 'express';
import { uploadEyeImage, getUserEyeImages, getLatestEyeImage, deleteEyeImage } from '../controllers/upload.controller.js';
import upload from '../middlewares/upload.middleware.js';

const router = express.Router();

// Upload endpoints
router.post('/eye-image', upload.single('image'), uploadEyeImage);
router.get('/eye-images/:username', getUserEyeImages);
router.get('/eye-image/latest/:username', getLatestEyeImage);
router.delete('/eye-image/:imageId', deleteEyeImage);

export default router;