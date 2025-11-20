"""
Minimal Flask Startup - Bypasses encoding issues
"""
import os
import sys

# Suppress ALL output during model loading
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# Set encoding before imports
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Starting Flask Backend...")

# Import and load model with suppressed output
with SuppressOutput():
    from app import app, initialize_model
    model_loaded = initialize_model()

if model_loaded:
    print("Model loaded successfully!")
else:
    print("Warning: Model failed to load, but starting server anyway...")

print("\nFlask server starting on http://localhost:5000")
print("Press CTRL+C to stop\n")

# Start Flask
app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
