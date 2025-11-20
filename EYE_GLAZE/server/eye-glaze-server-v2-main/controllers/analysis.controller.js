import { getDb } from '../configs/db.config.js';

// Submit stress analysis result
export const submitAnalysis = async (req, res) => {
  try {
    const { username, hasStress, imageUrl } = req.body;

    // Validate request
    if (!username || hasStress === undefined) {
      return res.status(400).json({
        status: 'error',
        message: 'Username and stress status (hasStress) are required'
      });
    }

    // Ensure hasStress is a boolean
    if (typeof hasStress !== 'boolean') {
      return res.status(400).json({
        status: 'error',
        message: 'hasStress must be a boolean value (true or false)'
      });
    }

    const db = getDb();
    const analysisCollection = db.collection('analyses');
    const usersCollection = db.collection('users');

    // Check if user exists
    const user = await usersCollection.findOne({ username });
    if (!user) {
      return res.status(404).json({
        status: 'error',
        message: 'User not found'
      });
    }

    // Create new analysis record
    const newAnalysis = {
      username,
      hasStress,
      imageUrl: imageUrl || null, // Store the imageUrl if provided, otherwise null
      createdAt: new Date()
    };

    const result = await analysisCollection.insertOne(newAnalysis);

    res.status(201).json({
      status: 'success',
      message: 'Analysis submitted successfully',
      data: {
        id: result.insertedId,
        username: newAnalysis.username,
        hasStress: newAnalysis.hasStress,
        imageUrl: newAnalysis.imageUrl,
        createdAt: newAnalysis.createdAt
      }
    });
  } catch (error) {
    console.error('Error submitting analysis:', error);
    res.status(500).json({
      status: 'error',
      message: 'Internal server error'
    });
  }
};

// Get all analyses for a specific user
export const getUserAnalyses = async (req, res) => {
  try {
    const { username } = req.params;

    if (!username) {
      return res.status(400).json({
        status: 'error',
        message: 'Username is required'
      });
    }

    const db = getDb();
    const analysisCollection = db.collection('analyses');

    // Find all analyses for the user, sorted by creation date (newest first)
    const analyses = await analysisCollection
      .find({ username })
      .sort({ createdAt: -1 })
      .toArray();

    res.status(200).json({
      status: 'success',
      message: 'User analyses retrieved successfully',
      count: analyses.length,
      data: analyses
    });
  } catch (error) {
    console.error('Error retrieving analyses:', error);
    res.status(500).json({
      status: 'error',
      message: 'Internal server error'
    });
  }
};

// Get latest analysis for a user
export const getLatestAnalysis = async (req, res) => {
  try {
    const { username } = req.params;

    if (!username) {
      return res.status(400).json({
        status: 'error',
        message: 'Username is required'
      });
    }

    const db = getDb();
    const analysisCollection = db.collection('analyses');

    // Find the most recent analysis for the user
    const latestAnalysis = await analysisCollection
      .find({ username })
      .sort({ createdAt: -1 })
      .limit(1)
      .toArray();

    if (latestAnalysis.length === 0) {
      return res.status(404).json({
        status: 'error',
        message: 'No analysis found for this user'
      });
    }

    res.status(200).json({
      status: 'success',
      message: 'Latest analysis retrieved successfully',
      data: latestAnalysis[0]
    });
  } catch (error) {
    console.error('Error retrieving latest analysis:', error);
    res.status(500).json({
      status: 'error',
      message: 'Internal server error'
    });
  }
};

// Get analysis count for a specific user
export const getUserAnalysisCount = async (req, res) => {
  try {
    const { username } = req.params;

    if (!username) {
      return res.status(400).json({
        status: 'error',
        message: 'Username is required'
      });
    }

    const db = getDb();
    const analysisCollection = db.collection('analyses');
    const usersCollection = db.collection('users');

    // Check if user exists
    const user = await usersCollection.findOne({ username });
    if (!user) {
      return res.status(404).json({
        status: 'error',
        message: 'User not found'
      });
    }

    // Count all analyses for this user
    const count = await analysisCollection.countDocuments({ username });

    res.status(200).json({
      status: 'success',
      message: 'Analysis count retrieved successfully',
      data: {
        username,
        totalAnalyses: count
      }
    });
  } catch (error) {
    console.error('Error retrieving analysis count:', error);
    res.status(500).json({
      status: 'error',
      message: 'Internal server error'
    });
  }
};