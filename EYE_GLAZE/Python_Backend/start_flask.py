"""
Simple Flask startup script without Unicode characters
"""
import sys
import os

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'backslashreplace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'backslashreplace')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

print("="*80)
print("INITIALIZING FLASK BACKEND")
print("="*80)
print("Loading model... (this may take 10-15 seconds)")
print("")

# Import and run the app
from app import app, initialize_model

if __name__ == '__main__':
    print("Starting model initialization...")
    
    try:
        success = initialize_model()
        
        if not success:
            print("[WARNING] Model not loaded. Server will start but predictions will fail.")
        else:
            print("[SUCCESS] Model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Initialization error: {e}")
    
    print("")
    print("="*80)
    print("Starting Flask server on http://localhost:5000")
    print("Accepting requests from React frontend (port 5173)")
    print("Press CTRL+C to stop")
    print("="*80)
    print("")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Disable debug mode to reduce output
        threaded=True
    )
