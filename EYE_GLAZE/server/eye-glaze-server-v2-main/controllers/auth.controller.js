import { getDb } from '../configs/db.config.js';

// Login user
export const loginUser = async (req, res) => {
  try {
    const { username, password } = req.body;

    // Validate request
    if (!username || !password) {
      return res.status(400).json({
        status: 'error',
        message: 'Username and password are required'
      });
    }

    const db = getDb();
    const usersCollection = db.collection('users');

    // Check if user exists and password matches
    const user = await usersCollection.findOne({ username });
    
    if (!user) {
      return res.status(401).json({
        status: 'error',
        message: 'Invalid username or password'
      });
    }

    // Check if password matches
    const passwordMatches = user.password === password;
    
    if (!passwordMatches) {
      return res.status(401).json({
        status: 'error',
        message: 'Invalid username or password'
      });
    }

    // Calculate age from birth date
    const birthDate = new Date(user.birthDate);
    const age = Math.floor((new Date() - birthDate) / (365.25 * 24 * 60 * 60 * 1000));

    // Return user data (excluding password)
    const { password: _, birthDate: __, ...userWithoutSensitive } = user;
    
    res.status(200).json({
      status: 'success',
      message: 'Login successful',
      data: {
        ...userWithoutSensitive,
        age
      }
    });
  } catch (error) {
    console.error('Error logging in:', error);
    res.status(500).json({
      status: 'error',
      message: 'Internal server error'
    });
  }
};

// Register a new user
export const registerUser = async (req, res) => {
  try {
    const { username, password, birthDate } = req.body;

    // Validate request
    if (!username || !password || !birthDate) {
      return res.status(400).json({
        status: 'error',
        message: 'Username, password, and birth date are required'
      });
    }

    // Validate birth date format
    const birthDateObj = new Date(birthDate);
    if (isNaN(birthDateObj.getTime())) {
      return res.status(400).json({
        status: 'error',
        message: 'Invalid birth date format. Please use YYYY-MM-DD format'
      });
    }

    const db = getDb();
    const usersCollection = db.collection('users');

    // Check if user already exists
    const existingUser = await usersCollection.findOne({ username });
    if (existingUser) {
      return res.status(409).json({
        status: 'error',
        message: 'Username already exists'
      });
    }

    // Create new user
    const newUser = {
      username,
      password, // Note: In a production app, you would hash this password
      birthDate: birthDateObj,
      createdAt: new Date()
    };

    const result = await usersCollection.insertOne(newUser);

    // Calculate age
    const age = Math.floor((new Date() - birthDateObj) / (365.25 * 24 * 60 * 60 * 1000));

    res.status(201).json({
      status: 'success',
      message: 'User registered successfully',
      data: {
        id: result.insertedId,
        username: newUser.username,
        age,
        createdAt: newUser.createdAt
      }
    });
  } catch (error) {
    console.error('Error registering user:', error);
    res.status(500).json({
      status: 'error',
      message: 'Internal server error'
    });
  }
};
