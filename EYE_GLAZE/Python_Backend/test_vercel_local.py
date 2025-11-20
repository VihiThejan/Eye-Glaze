"""
Test the Vercel serverless function locally
"""
import sys
import os

# Add api directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

try:
    from index import app
    print("✅ Successfully imported Flask app from api/index.py")
    
    # Test that the app is a Flask instance
    from flask import Flask
    if isinstance(app, Flask):
        print("✅ App is a valid Flask instance")
    else:
        print("❌ App is not a Flask instance")
        sys.exit(1)
    
    # List routes
    print("\n📋 Available routes:")
    for rule in app.url_map.iter_rules():
        methods = ','.join(rule.methods - {'HEAD', 'OPTIONS'})
        print(f"  {rule.rule:30s} [{methods}]")
    
    print("\n✅ All checks passed!")
    print("\nTo run locally:")
    print("  python api/index.py")
    print("  Then visit: http://localhost:5000")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
