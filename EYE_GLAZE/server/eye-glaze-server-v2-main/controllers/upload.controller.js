import { getDb } from '../configs/db.config.js';
import cloudinary from '../configs/cloudinary.config.js';
import fs from 'fs';

// Upload eye image
export const uploadEyeImage = async (req, res) => {
  try {
    const { username } = req.body;
    
    // Validate request
    if (!username) {
      return res.status(400).json({
        status: 'error',
        message: 'Username is required'
      });
    }
    
    // Validate file upload
    if (!req.file) {
      return res.status(400).json({
        status: 'error',
        message: 'No image file uploaded'
      });
    }
    
    const db = getDb();
    const usersCollection = db.collection('users');
    const eyeImagesCollection = db.collection('eye_images');
    
    // Check if user exists
    const user = await usersCollection.findOne({ username });
    if (!user) {
      return res.status(404).json({
        status: 'error',
        message: 'User not found'
      });
    }
    
    try {
      // Upload image to Cloudinary
      const result = await cloudinary.uploader.upload(req.file.path, {
        folder: 'eye_glaze_images',
        resource_type: 'image'
      });
      
      // Remove temporary file after upload
      fs.unlinkSync(req.file.path);
      
      // Create new eye image record
      const newEyeImage = {
        username,
        imageUrl: result.secure_url,
        cloudinaryId: result.public_id,
        uploadedAt: new Date()
      };
      
      // Save to database
      const dbResult = await eyeImagesCollection.insertOne(newEyeImage);
      
      res.status(201).json({
        status: 'success',
        message: 'Eye image uploaded successfully',
        data: {
          id: dbResult.insertedId,
          username: newEyeImage.username,
          imageUrl: newEyeImage.imageUrl,
          uploadedAt: newEyeImage.uploadedAt
        }
      });
    } catch (uploadError) {
      // Remove temporary file if upload fails
      if (req.file && req.file.path) {
        fs.unlinkSync(req.file.path);
      }
      
      console.error('Error uploading to Cloudinary:', uploadError);
      return res.status(500).json({
        status: 'error',
        message: 'Failed to upload image',
        error: uploadError.message
      });
    }
  } catch (error) {
    console.error('Error uploading eye image:', error);
    res.status(500).json({
      status: 'error',
      message: 'Internal server error'
    });
  }
};

// Get all eye images for a specific user
export const getUserEyeImages = async (req, res) => {
  try {
    const { username } = req.params;
    
    if (!username) {
      return res.status(400).json({
        status: 'error',
        message: 'Username is required'
      });
    }
    
    const db = getDb();
    const eyeImagesCollection = db.collection('eye_images');
    
    // Find all eye images for the user, sorted by upload date (newest first)
    const images = await eyeImagesCollection
      .find({ username })
      .sort({ uploadedAt: -1 })
      .toArray();
    
    res.status(200).json({
      status: 'success',
      message: 'User eye images retrieved successfully',
      count: images.length,
      data: images
    });
  } catch (error) {
    console.error('Error retrieving eye images:', error);
    res.status(500).json({
      status: 'error',
      message: 'Internal server error'
    });
  }
};

// Get latest eye image for a user
export const getLatestEyeImage = async (req, res) => {
  try {
    const { username } = req.params;
    
    if (!username) {
      return res.status(400).json({
        status: 'error',
        message: 'Username is required'
      });
    }
    
    const db = getDb();
    const eyeImagesCollection = db.collection('eye_images');
    
    // Find the most recent eye image for the user
    const latestImage = await eyeImagesCollection
      .find({ username })
      .sort({ uploadedAt: -1 })
      .limit(1)
      .toArray();
    
    if (latestImage.length === 0) {
      return res.status(404).json({
        status: 'error',
        message: 'No eye images found for this user'
      });
    }
    
    res.status(200).json({
      status: 'success',
      message: 'Latest eye image retrieved successfully',
      data: latestImage[0]
    });
  } catch (error) {
    console.error('Error retrieving latest eye image:', error);
    res.status(500).json({
      status: 'error',
      message: 'Internal server error'
    });
  }
};

// Delete an eye image
export const deleteEyeImage = async (req, res) => {
  try {
    const { imageId } = req.params;
    
    if (!imageId) {
      return res.status(400).json({
        status: 'error',
        message: 'Image ID is required'
      });
    }
    
    const db = getDb();
    const eyeImagesCollection = db.collection('eye_images');
    
    // Find the image to get the cloudinary ID
    const image = await eyeImagesCollection.findOne({ _id: imageId });
    
    if (!image) {
      return res.status(404).json({
        status: 'error',
        message: 'Image not found'
      });
    }
    
    // Delete from Cloudinary
    await cloudinary.uploader.destroy(image.cloudinaryId);
    
    // Delete from database
    await eyeImagesCollection.deleteOne({ _id: imageId });
    
    res.status(200).json({
      status: 'success',
      message: 'Eye image deleted successfully'
    });
  } catch (error) {
    console.error('Error deleting eye image:', error);
    res.status(500).json({
      status: 'error',
      message: 'Internal server error'
    });
  }
};