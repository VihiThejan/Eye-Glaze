"""
Safe Flask Launcher with UTF-8 encoding enforcement
Fixes Windows PowerShell cp1252 encoding issues
"""

import os
import sys
import codecs

# Force UTF-8 encoding for all I/O
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'backslashreplace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'backslashreplace')

# Set environment variables
os.environ['PYTHONIOENCODING'] = 'utf-8:backslashreplace'
os.environ['PYTHONUTF8'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Now import and run Flask app
if __name__ == '__main__':
    import app
    
    print("[INFO] Starting Flask with UTF-8 encoding...")
    
    # Initialize model
    if not app.initialize_model():
        print("[WARNING] Model not loaded. Server will start but predictions will fail.")
    
    # Start Flask server
    print("\nStarting Flask server on http://localhost:5000")
    print("Accepting requests from React frontend (port 5173)")
    print("Press CTRL+C to stop\n")
    
    app.app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Disable debug to avoid auto-reload encoding issues
        threaded=True
    )
