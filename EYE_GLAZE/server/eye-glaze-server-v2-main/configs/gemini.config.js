import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';

dotenv.config();

// Initialize the Google Generative AI with your API key
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// Use the Gemini model
const model = {
  generateContent: async (prompt) => {
    const result = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt
    });
    
    return {
      response: {
        text: () => result.text
      }
    };
  }
};

export { ai, model };