# Eye-Glaze Server

A Node.js Express server with MongoDB integration for eye and stress analysis platform, featuring user authentication and stress analysis tracking.

## Features

- User registration and login
- Stress analysis submission and retrieval
- Eye image upload and management with Cloudinary
- AI-powered recommendations using Google Gemini API
- Eye health summary and statistics
- MongoDB integration for data storage
- Health and database connection monitoring
- RESTful API structure

## Prerequisites

- [Node.js](https://nodejs.org/) (v16+ recommended)
- [MongoDB Atlas](https://www.mongodb.com/atlas/database) account or local MongoDB installation
- [npm](https://www.npmjs.com/) or [yarn](https://yarnpkg.com/)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Derikklok/eye-glaze-server.git
   cd eye-glaze-server/server-1
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create a `.env` file in the root directory with the following content:
   ```
   EXPRESS_PORT=5174
   EXPRESS_MONGO_NODE_SRV=your_mongodb_connection_string
   ```
   Replace `your_mongodb_connection_string` with your actual MongoDB connection string.

## Running the Application

### Development Mode

Run the server in development mode with hot reloading:
```bash
npm run dev
```

### Production Mode

Start the server in production mode:
```bash
npm start
```

## API Endpoints

### Health & Status

- `GET /` - Simple health check
- `GET /api/health` - Detailed health and status information
- `GET /api/status/db` - MongoDB connection status

### Authentication

- `POST /api/auth/register` - Register a new user
  ```json
  // Request
  {
    "username": "john_doe",
    "password": "secure_password",
    "birthDate": "1990-05-15"  // Required, format: YYYY-MM-DD
  }

  // Response
  {
    "status": "success",
    "message": "User registered successfully",
    "data": {
      "id": "507f1f77bcf86cd799439011",
      "username": "john_doe",
      "age": 35,
      "createdAt": "2025-11-09T10:30:00.000Z"
    }
  }
  ```

- `POST /api/auth/login` - Login with existing credentials
  ```json
  // Request
  {
    "username": "john_doe",
    "password": "secure_password"
  }

  // Response
  {
    "status": "success",
    "message": "Login successful",
    "data": {
      "id": "507f1f77bcf86cd799439011",
      "username": "john_doe",
      "age": 35,
      "createdAt": "2025-11-09T10:30:00.000Z"
    }
  }
  ```

- `GET /api/auth/user` - Get user information (sample route)

### Analysis

- `POST /api/analysis/submit` - Submit a stress analysis result
  ```json
  {
    "username": "your_username",
    "hasStress": true,
    "imageUrl": "https://res.cloudinary.com/yourcloud/image/upload/v1698765432/eye_glaze_images/image-1698765432123.jpg"
  }
  ```
  Response:
  ```json
  {
    "status": "success",
    "message": "Analysis submitted successfully",
    "data": {
      "id": "655d4b3fc9e5c123456789ab",
      "username": "your_username",
      "hasStress": true,
      "imageUrl": "https://res.cloudinary.com/yourcloud/image/upload/v1698765432/eye_glaze_images/image-1698765432123.jpg",
      "createdAt": "2025-10-13T15:45:03.789Z"
    }
  }
  ```

- `GET /api/analysis/user/:username` - Get all analyses for a specific user (includes imageUrl)
  
  Response:
  ```json
  {
    "status": "success",
    "message": "User analyses retrieved successfully",
    "count": 2,
    "data": [
      {
        "_id": "655d4b3fc9e5c123456789ab",
        "username": "your_username",
        "hasStress": true,
        "imageUrl": "https://res.cloudinary.com/yourcloud/image/upload/v1698765432/eye_glaze_images/image-1698765432123.jpg",
        "createdAt": "2025-10-13T15:45:03.789Z"
      },
      {
        "_id": "655d1c5ec9e5c123456789cd",
        "username": "your_username",
        "hasStress": false,
        "imageUrl": null,
        "createdAt": "2025-10-13T12:20:30.456Z"
      }
    ]
  }
  ```

### Eye Image Upload

- `POST /api/upload/eye-image` - Upload an eye image for analysis
  ```
  Form-data:
  - username: "your_username"
  - image: [file upload]
  ```
  Response:
  ```json
  {
    "status": "success",
    "message": "Eye image uploaded successfully",
    "data": {
      "id": "655e7d2fc9e5c123456789ef",
      "username": "your_username",
      "imageUrl": "https://res.cloudinary.com/yourcloud/image/upload/v1698765432/eye_glaze_images/image-1698765432123.jpg",
      "uploadedAt": "2025-10-13T20:17:12.123Z"
    }
  }
  ```

- `GET /api/upload/eye-images/:username` - Get all eye images for a specific user
  
  Response:
  ```json
  {
    "status": "success",
    "message": "User eye images retrieved successfully",
    "count": 2,
    "data": [
      {
        "_id": "655e7d2fc9e5c123456789ef",
        "username": "your_username",
        "imageUrl": "https://res.cloudinary.com/yourcloud/image/upload/v1698765432/eye_glaze_images/image-1698765432123.jpg",
        "cloudinaryId": "eye_glaze_images/image-1698765432123",
        "uploadedAt": "2025-10-13T20:17:12.123Z"
      },
      {
        "_id": "655e6b1ec9e5c123456789gh",
        "username": "your_username",
        "imageUrl": "https://res.cloudinary.com/yourcloud/image/upload/v1698764321/eye_glaze_images/image-1698764321456.jpg",
        "cloudinaryId": "eye_glaze_images/image-1698764321456",
        "uploadedAt": "2025-10-13T19:45:21.456Z"
      }
    ]
  }
  ```

- `GET /api/analysis/latest/:username` - Get the most recent analysis for a user (includes imageUrl)
  
  Response:
  ```json
  {
    "status": "success",
    "message": "Latest analysis retrieved successfully",
    "data": {
      "_id": "655d4b3fc9e5c123456789ab",
      "username": "your_username",
      "hasStress": true,
      "imageUrl": "https://res.cloudinary.com/yourcloud/image/upload/v1698765432/eye_glaze_images/image-1698765432123.jpg",
      "createdAt": "2025-10-13T15:45:03.789Z"
    }
  }
  ```

- `GET /api/analysis/count/:username` - Get the total count of analyses submitted by a user
  
  Response:
  ```json
  {
    "status": "success",
    "message": "Analysis count retrieved successfully",
    "data": {
      "username": "your_username",
      "totalAnalyses": 5
    }
  }
  ```

- `GET /api/upload/eye-image/latest/:username` - Get the most recent eye image for a user
  
  Response:
  ```json
  {
    "status": "success",
    "message": "Latest eye image retrieved successfully",
    "data": {
      "_id": "655e7d2fc9e5c123456789ef",
      "username": "your_username",
      "imageUrl": "https://res.cloudinary.com/yourcloud/image/upload/v1698765432/eye_glaze_images/image-1698765432123.jpg",
      "cloudinaryId": "eye_glaze_images/image-1698765432123",
      "uploadedAt": "2025-10-13T20:17:12.123Z"
    }
  }
  ```

- `DELETE /api/upload/eye-image/:imageId` - Delete a specific eye image
  
  Response:
  ```json
  {
    "status": "success",
    "message": "Eye image deleted successfully"
  }
  ```

### AI-Powered Recommendations

- `GET /api/recommendations/user/:username` - Get personalized AI recommendations based on user's eye analysis
  
  Response:
  ```json
  {
    "status": "success",
    "message": "Recommendations generated successfully",
    "data": {
      "analysis": {
        "hasStress": true,
        "createdAt": "2025-10-13T15:45:03.789Z",
        "imageUrl": "https://res.cloudinary.com/yourcloud/image/upload/v1698765432/eye_glaze_images/image-1698765432123.jpg"
      },
      "stats": {
        "totalAnalysesLastWeek": 5,
        "stressDetectedCount": 3,
        "stressPercentage": 60
      },
      "recommendations": {
        "assessment": "Based on your recent eye analyses, you're experiencing moderate eye strain. In the past week, stress was detected in 60% of your analyses, which suggests your eyes may be overworked.",
        "recommendations": [
          "Take regular 20-20-20 breaks (look 20 feet away for 20 seconds every 20 minutes)",
          "Adjust screen brightness to match your surroundings",
          "Consider using blue light filtering glasses during extended screen time",
          "Position your screen slightly below eye level and at arm's length away",
          "Increase font size to reduce eye strain when reading"
        ],
        "lifestyleAdjustments": [
          "Stay hydrated throughout the day",
          "Ensure proper lighting in your workspace",
          "Get sufficient sleep (7-8 hours) to help eyes recover",
          "Practice eye exercises like palming and focus shifting"
        ]
      }
    }
  }
  ```

- `GET /api/recommendations/summary/:username` - Get a statistical summary of user's eye health history
  
  Response:
  ```json
  {
    "status": "success",
    "message": "Eye health summary retrieved successfully",
    "data": {
      "summary": {
        "totalAnalyses": 15,
        "stressDetectedCount": 7,
        "stressPercentage": 47,
        "latestStatus": "Stress detected",
        "latestAnalysisTime": "2025-10-13T15:45:03.789Z"
      },
      "weeklyTrends": [
        {
          "week": "2025-10-6",
          "total": 3,
          "stressDetected": 1,
          "percentage": 33
        },
        {
          "week": "2025-10-13",
          "total": 5,
          "stressDetected": 3,
          "percentage": 60
        }
      ]
    }
  }
  ```

- `GET /api/recommendations/history/:username` - Get a user's saved recommendation history
  
  Response:
  ```json
  {
    "status": "success",
    "message": "Recommendations history retrieved successfully",
    "data": {
      "recommendations": [
        {
          "_id": "655f9c3fc9e5c123456789ij",
          "username": "your_username",
          "userId": "655c3a2ec9e5c123456789kl",
          "analysisId": "655d4b3fc9e5c123456789ab",
          "createdAt": "2025-10-14T10:20:15.678Z",
          "analysisData": {
            "hasStress": true,
            "createdAt": "2025-10-13T15:45:03.789Z",
            "imageUrl": "https://res.cloudinary.com/yourcloud/image/upload/v1698765432/eye_glaze_images/image-1698765432123.jpg"
          },
          "stats": {
            "totalAnalysesLastWeek": 5,
            "stressDetectedCount": 3,
            "stressPercentage": 60
          },
          "recommendations": {
            "assessment": "Based on your recent eye analyses, you're experiencing moderate eye strain...",
            "recommendations": [
              "Take regular 20-20-20 breaks (look 20 feet away for 20 seconds every 20 minutes)",
              "Adjust screen brightness to match your surroundings",
              "Consider using blue light filtering glasses during extended screen time"
            ],
            "lifestyleAdjustments": [
              "Stay hydrated throughout the day",
              "Ensure proper lighting in your workspace",
              "Get sufficient sleep (7-8 hours) to help eyes recover"
            ]
          }
        }
      ],
      "pagination": {
        "total": 5,
        "limit": 10,
        "skip": 0,
        "hasMore": false
      }
    }
  }
  ```

## Project Structure

```
server-1/
├── configs/          # Configuration files
│   ├── db.config.js        # MongoDB connection configuration
│   ├── cloudinary.config.js # Cloudinary configuration
│   └── gemini.config.js    # Google Gemini AI configuration
├── controllers/      # Request controllers
│   ├── auth.controller.js           # Authentication controller
│   ├── analysis.controller.js       # Analysis controller
│   ├── upload.controller.js         # Image upload controller
│   └── recommendations.controller.js # AI recommendations controller
├── middlewares/      # Middleware functions
│   └── upload.middleware.js   # File upload middleware
├── routes/           # API routes
│   ├── auth.routes.js           # Authentication routes
│   ├── analysis.routes.js       # Analysis routes
│   ├── upload.routes.js         # Upload routes
│   └── recommendations.routes.js # Recommendations routes
├── uploads/          # Temporary storage for uploaded files
├── .env              # Environment variables
├── .gitignore        # Git ignore file
├── index.js          # Application entry point
├── package.json      # Project dependencies
└── README.md         # Project documentation
```

## Environment Variables

- `EXPRESS_PORT` - Port the server will run on (default: 5176)
- `EXPRESS_MONGO_NODE_SRV` - MongoDB connection string
- `CLOUDINARY_CLOUD_NAME` - Cloudinary cloud name
- `CLOUDINARY_API_KEY` - Cloudinary API key
- `CLOUDINARY_API_SECRET` - Cloudinary API secret
- `GEMINI_API_KEY` - Google Gemini API key

## Development

### Available Scripts

- `npm start` - Run the server in production mode
- `npm run dev` - Run the server in development mode with hot reloading
- `npm test` - Run tests (not configured yet)

## Error Handling

The server includes a global error handler that catches and formats errors appropriately.

## Security Notes

This is a simple implementation without JWT authentication or password hashing. For production applications, consider implementing:

- Password hashing (using bcrypt or similar)
- JWT authentication
- Input validation
- Rate limiting
- HTTPS

## License

ISC
