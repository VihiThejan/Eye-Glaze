# Eye Glaze - Stress Detection System

Eye Glaze is a sophisticated web application that uses eye analysis to detect stress levels in users. Built with React, TypeScript, and Tailwind CSS, it provides a modern and intuitive interface for stress monitoring and management.

## ML Service Integration Guide

### Accessing User Age for ML Analysis

If you're working on the ML service integration and need to access the user's age for analysis, follow these steps:

#### Step 1: Access User Age
The user's age can be accessed in two ways:

1. **Direct localStorage Access:**
```typescript
// Method 1: Direct localStorage access
const getUserAge = () => {
  const storedUser = localStorage.getItem('eyeGlazeUser');
  if (storedUser) {
    const userData = JSON.parse(storedUser);
    return userData.age || null;
  }
  return null;
};
```

2. **Using AuthContext (Recommended for React Components):**
```typescript
import { useAuth } from '@/contexts/AuthContext';

function YourMLComponent() {
  const { user } = useAuth();
  const userAge = user?.age;
  // Use userAge in your ML logic
}
```

#### Step 2: Include Age in ML Request
When sending data to your ML service, include the age in the request:

```typescript
// Example: Adding age to ML service request
const sendToMLService = async (imageFile: File) => {
  const userAge = getUserAge(); // Get age using method above
  
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('age', userAge?.toString() || ''); // Add age to form data
  
  try {
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    return result;
  } catch (error) {
    console.error('ML Service error:', error);
    throw error;
  }
};
```

#### Step 3: Handle Age in Python Backend
In your Flask/Python ML service, access the age like this:

```python
from flask import request

@app.route('/predict', methods=['POST'])
def predict():
    # Get the age from the request
    age = request.form.get('age')
    
    # Convert to int if age is provided
    user_age = int(age) if age else None
    
    # Use age in your ML model
    if user_age:
        # Add age to your model's input features
        # Your ML logic here
        pass
    
    # Rest of your prediction code
    return jsonResponse
```

#### Important Notes:
1. Always validate age exists before using
2. Handle cases where age might be missing
3. Remember to convert age to string for FormData
4. Use try-catch blocks for error handling

## Features

### Core Features
- 👁️ Real-time eye image analysis
- 📊 Stress level detection with AI
- 📈 Historical stress data tracking
- 🎯 Personalized wellness recommendations
- 👤 User profile management with age tracking

### User Management
- 📝 User registration with age collection
- 🔐 Secure authentication system
- 👤 Age-aware user profiles
- 💾 Persistent user data storage

### Analysis Features
- 📸 Easy image upload (drag & drop supported)
- 🔍 Advanced eye pattern analysis
- 📊 Detailed stress probability metrics
- 📈 Historical trend analysis
- 🎯 Confidence level indicators

## Getting Started

### Prerequisites
- Node.js (v16 or higher)
- npm or yarn
- Python backend running on port 5174 (for API)

### Installation

1. Clone the repository:
\`\`\`bash
git clone https://github.com/your-username/eye-glaze-client-react-vite-main.git
cd eye-glaze-client-react-vite-main
\`\`\`

2. Install dependencies:
\`\`\`bash
npm install
\`\`\`

3. Start the development server:
\`\`\`bash
npm run dev
\`\`\`

## Working with User Data

### Accessing User Age from localStorage

The user's age is stored in localStorage along with other user data. Here's how to access it:

\`\`\`typescript
// Reading user data including age
const getUserData = () => {
  const storedUser = localStorage.getItem('eyeGlazeUser');
  if (storedUser) {
    const userData = JSON.parse(storedUser);
    const userAge = userData.age; // Access the age property
    return userData;
  }
  return null;
};

// Example usage
const userData = getUserData();
if (userData?.age) {
  console.log(\`User's age: \${userData.age}\`);
}
\`\`\`

### User Data Structure in localStorage

The user data is stored in localStorage under the key 'eyeGlazeUser' with the following structure:

\`\`\`typescript
interface User {
  id: string;
  email: string;
  name: string;
  age?: number;
}
\`\`\`

## Authentication Context

The application uses a React Context (AuthContext) for managing user authentication and data. The context provides:

- User data including age
- Login functionality
- Registration with age collection
- Logout functionality
- Loading state management

Example usage in components:

\`\`\`typescript
import { useAuth } from '@/contexts/AuthContext';

function YourComponent() {
  const { user } = useAuth();
  
  return (
    <div>
      {user?.age && <p>Age: {user.age} years</p>}
    </div>
  );
}
\`\`\`

## Development

### Running the Full Stack

To run both the frontend and backend servers:

\`\`\`bash
npm run dev:full
\`\`\`

This will start:
- React frontend on default port (usually 5173)
- Python backend on port 5174
- Flask ML service on port 5000

### Environment Variables

No additional environment variables are required for the basic setup. The backend URLs are hardcoded to:
- Main API: http://localhost:5174
- ML Service: http://localhost:5000

## Contributing

1. Fork the repository
2. Create your feature branch (\`git checkout -b feature/AmazingFeature\`)
3. Commit your changes (\`git commit -m 'Add some AmazingFeature'\`)
4. Push to the branch (\`git push origin feature/AmazingFeature\`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
