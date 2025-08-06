#!/usr/bin/env python3
"""
Setup script for PolicyQuery AI
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("=" * 60)
    print("🚀 PolicyQuery AI Setup")
    print("=" * 60)
    print("LLM-Powered Intelligent Query-Retrieval System")
    print("Built for Bajaj Finserv Hackathon")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_env_file():
    """Create .env file from template"""
    print("\n🔧 Setting up environment variables...")
    
    env_file = Path(".env")
    env_example = Path("env_example.txt")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if not env_example.exists():
        print("❌ env_example.txt not found")
        return False
    
    try:
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file with your API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def validate_setup():
    """Validate the setup"""
    print("\n🔍 Validating setup...")
    
    # Check if main files exist
    required_files = ["main.py", "requirements.txt", "README.md"]
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    # Check if .env exists
    if not Path(".env").exists():
        print("⚠️  .env file not found. Please create it with your API keys")
    
    print("✅ Setup validation completed")
    return True

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("🎉 Setup completed!")
    print("=" * 60)
    print("\n📋 Next steps:")
    print("1. Edit .env file with your API keys:")
    print("   - GEMINI_API_KEY: Get from https://makersuite.google.com/app/apikey")
    print("   - PINECONE_API_KEY: Get from https://app.pinecone.io/")
    print("\n2. Run the application:")
    print("   python main.py")
    print("   # or")
    print("   python main_enhanced.py")
    print("\n3. Test the API:")
    print("   python test_api.py")
    print("\n4. Access API documentation:")
    print("   http://localhost:8000/docs")
    print("\n5. Health check:")
    print("   http://localhost:8000/health")
    print("\n" + "=" * 60)

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create environment file
    create_env_file()
    
    # Validate setup
    if not validate_setup():
        sys.exit(1)
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main() 