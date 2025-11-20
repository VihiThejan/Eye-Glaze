import express from 'express'
import { registerUser, loginUser } from '../controllers/auth.controller.js';

const router = express.Router();

// Authentication endpoints
router.post('/register', registerUser);
router.post('/login', loginUser);

// Existing endpoint
router.get('/user', (req, res) => {
    console.log('user captured');
    res.status(200).json({ message: 'User route' });
});

export default router;