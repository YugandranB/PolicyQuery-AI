#!/usr/bin/env python3
"""
Environment setup script for PolicyQuery AI
"""

import os
from pathlib import Path

def create_env_file():
    """Create .env file with the provided API keys"""
    
    env_content = """# API Keys
GEMINI_API_KEY=AIzaSyAs89oFf8MY2mRvYeVAS2BpAIDyZSMHObQ
PINECONE_API_KEY=pcsk_6dq3tu_UweY8AdAYSCF9WKz3JoCjNeMFYneesmYwh6jERQ7rV8y9J1HLSQU1t52QxXVyeX
PINECONE_ENVIRONMENT=gcp-starter

# Server Configuration
HOST=0.0.0.0
PORT=8000

# Authentication
AUTH_TOKEN=3ac5ba63774c0a827bb0436425359a4bc2519da9f4da61af84b5685a1237d4bc
"""
    
    env_file = Path(".env")
    
    try:
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ Created .env file with your API keys")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def validate_env_setup():
    """Validate that environment variables are set"""
    print("\n🔍 Validating environment setup...")
    
    required_vars = [
        "GEMINI_API_KEY",
        "PINECONE_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("Please run this script to set them up:")
        print("python setup_env.py")
        return False
    
    print("✅ Environment variables are properly configured")
    return True

def main():
    """Main setup function"""
    print("🔧 Setting up environment variables for PolicyQuery AI")
    print("=" * 60)
    
    # Create .env file
    if create_env_file():
        print("\n📋 Environment variables configured:")
        print("  - GEMINI_API_KEY: ✅ Set")
        print("  - PINECONE_API_KEY: ✅ Set")
        print("  - PINECONE_ENVIRONMENT: gcp-starter")
        print("  - AUTH_TOKEN: ✅ Set")
        
        print("\n🚀 You can now run the application:")
        print("  python main.py")
        print("  # or")
        print("  python main_enhanced.py")
        
        print("\n🧪 Test the API:")
        print("  python test_api.py")
        
        print("\n📚 Access documentation:")
        print("  http://localhost:8000/docs")
        
    else:
        print("❌ Failed to set up environment variables")

if __name__ == "__main__":
    main() 