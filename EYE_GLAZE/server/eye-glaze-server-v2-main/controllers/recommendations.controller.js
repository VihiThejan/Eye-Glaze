import { getDb } from '../configs/db.config.js';
import { model } from '../configs/gemini.config.js';

// Get AI recommendations based on a user's analysis results
export const getRecommendations = async (req, res) => {
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
    
    // Get the most recent analysis for the user
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
    
    const analysis = latestAnalysis[0];
    
    // Count stress occurrences in the last week
    const oneWeekAgo = new Date();
    oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);
    
    const recentAnalyses = await analysisCollection
      .find({ 
        username,
        createdAt: { $gte: oneWeekAgo }
      })
      .toArray();
    
    const stressCount = recentAnalyses.filter(a => a.hasStress).length;
    const totalCount = recentAnalyses.length;
    const stressPercentage = totalCount > 0 ? Math.round((stressCount / totalCount) * 100) : 0;
    
    // Prepare prompt for Gemini AI
    const prompt = `
      You are an eye health advisor providing personalized recommendations based on eye stress analysis.
      
      User data:
      - Current stress detection: ${analysis.hasStress ? 'Stress detected' : 'No stress detected'}
      - Stress frequency in the past week: ${stressCount} out of ${totalCount} analyses (${stressPercentage}%)
      
      Based on this information, please provide:
      1. A brief assessment of the user's eye health situation
      2. 3-5 practical recommendations to improve their eye health
      3. Suggested lifestyle adjustments if they are showing signs of eye strain
      
      Format your response in JSON with these fields: 
      - assessment: A paragraph summarizing their situation
      - recommendations: An array of recommendation strings
      - lifestyleAdjustments: An array of adjustment suggestions
    `;
    
    // Generate content with Gemini AI
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const aiResponse = response.text();
    
    // Parse JSON from AI response
    let recommendationsData;
    try {
      // Extract JSON content if the AI wraps it in markdown code blocks
      const jsonMatch = aiResponse.match(/```json\s*([\s\S]*?)\s*```/) || 
                       aiResponse.match(/```\s*([\s\S]*?)\s*```/) ||
                       [null, aiResponse];
      
      const jsonContent = jsonMatch[1].trim();
      recommendationsData = JSON.parse(jsonContent);
    } catch (parseError) {
      console.error('Error parsing AI response:', parseError);
      // Fallback if JSON parsing fails
      recommendationsData = {
        assessment: "We couldn't analyze your eye health data properly. Please try again later.",
        recommendations: ["Take regular breaks from screen time", "Practice the 20-20-20 rule", "Ensure proper lighting"],
        lifestyleAdjustments: ["Maintain good posture", "Stay hydrated", "Ensure adequate sleep"]
      };
    }
    
    // Save recommendation to database
    const recommendationsCollection = db.collection('recommendations');
    
    const recommendationDocument = {
      username,
      userId: user._id,
      analysisId: analysis._id,
      createdAt: new Date(),
      analysisData: {
        hasStress: analysis.hasStress,
        createdAt: analysis.createdAt,
        imageUrl: analysis.imageUrl || null
      },
      stats: {
        totalAnalysesLastWeek: totalCount,
        stressDetectedCount: stressCount,
        stressPercentage: stressPercentage
      },
      recommendations: recommendationsData
    };
    
    // Insert the recommendation into the database
    await recommendationsCollection.insertOne(recommendationDocument);
    
    // Send response
    res.status(200).json({
      status: 'success',
      message: 'Recommendations generated successfully',
      data: {
        analysis: {
          hasStress: analysis.hasStress,
          createdAt: analysis.createdAt,
          imageUrl: analysis.imageUrl || null
        },
        stats: {
          totalAnalysesLastWeek: totalCount,
          stressDetectedCount: stressCount,
          stressPercentage: stressPercentage
        },
        recommendations: recommendationsData
      }
    });
  } catch (error) {
    console.error('Error generating recommendations:', error);
    res.status(500).json({
      status: 'error',
      message: 'Error generating recommendations',
      error: error.message
    });
  }
};

// Get summary of user's eye health history
export const getEyeHealthSummary = async (req, res) => {
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
    
    // Get all analyses for the user
    const allAnalyses = await analysisCollection
      .find({ username })
      .sort({ createdAt: -1 })
      .toArray();
    
    if (allAnalyses.length === 0) {
      return res.status(404).json({
        status: 'error',
        message: 'No analysis history found for this user'
      });
    }
    
    // Calculate summary statistics
    const totalAnalyses = allAnalyses.length;
    const stressDetectedCount = allAnalyses.filter(a => a.hasStress).length;
    const stressPercentage = Math.round((stressDetectedCount / totalAnalyses) * 100);
    
    // Group by week
    const weeklyData = {};
    allAnalyses.forEach(analysis => {
      const date = new Date(analysis.createdAt);
      const weekStart = new Date(date);
      weekStart.setDate(date.getDate() - date.getDay());
      const weekKey = `${weekStart.getFullYear()}-${weekStart.getMonth()+1}-${weekStart.getDate()}`;
      
      if (!weeklyData[weekKey]) {
        weeklyData[weekKey] = {
          week: weekKey,
          total: 0,
          stressDetected: 0,
          percentage: 0
        };
      }
      
      weeklyData[weekKey].total += 1;
      if (analysis.hasStress) {
        weeklyData[weekKey].stressDetected += 1;
      }
    });
    
    // Calculate percentages for each week
    Object.keys(weeklyData).forEach(key => {
      const week = weeklyData[key];
      week.percentage = Math.round((week.stressDetected / week.total) * 100);
    });
    
    // Convert to array and sort by week
    const weeklyTrends = Object.values(weeklyData).sort((a, b) => {
      return new Date(a.week) - new Date(b.week);
    });
    
    res.status(200).json({
      status: 'success',
      message: 'Eye health summary retrieved successfully',
      data: {
        summary: {
          totalAnalyses,
          stressDetectedCount,
          stressPercentage,
          latestStatus: allAnalyses[0].hasStress ? 'Stress detected' : 'No stress detected',
          latestAnalysisTime: allAnalyses[0].createdAt
        },
        weeklyTrends
      }
    });
  } catch (error) {
    console.error('Error generating eye health summary:', error);
    res.status(500).json({
      status: 'error',
      message: 'Error generating eye health summary',
      error: error.message
    });
  }
};

// Get user's saved recommendations history
export const getUserRecommendationHistory = async (req, res) => {
  try {
    const { username } = req.params;
    const { limit = 10, skip = 0 } = req.query;
    
    if (!username) {
      return res.status(400).json({
        status: 'error',
        message: 'Username is required'
      });
    }
    
    const db = getDb();
    const recommendationsCollection = db.collection('recommendations');
    const usersCollection = db.collection('users');
    
    // Check if user exists
    const user = await usersCollection.findOne({ username });
    if (!user) {
      return res.status(404).json({
        status: 'error',
        message: 'User not found'
      });
    }
    
    // Convert query params to numbers
    const numLimit = parseInt(limit);
    const numSkip = parseInt(skip);
    
    // Get recommendations history for user
    const recommendations = await recommendationsCollection
      .find({ username })
      .sort({ createdAt: -1 })
      .skip(numSkip)
      .limit(numLimit)
      .toArray();
    
    // Get total count for pagination
    const totalCount = await recommendationsCollection.countDocuments({ username });
    
    if (recommendations.length === 0) {
      return res.status(404).json({
        status: 'error',
        message: 'No recommendations found for this user'
      });
    }
    
    res.status(200).json({
      status: 'success',
      message: 'Recommendations history retrieved successfully',
      data: {
        recommendations,
        pagination: {
          total: totalCount,
          limit: numLimit,
          skip: numSkip,
          hasMore: numSkip + recommendations.length < totalCount
        }
      }
    });
  } catch (error) {
    console.error('Error retrieving recommendations history:', error);
    res.status(500).json({
      status: 'error',
      message: 'Error retrieving recommendations history',
      error: error.message
    });
  }
};