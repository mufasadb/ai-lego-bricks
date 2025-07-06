#!/bin/bash

# Agent Workflow Visualizer Startup Script
# This script starts the web interface for visualizing JSON agent workflows

echo "🔍 Starting Agent Workflow Visualizer..."
echo "================================================"

# Check if we're in the right directory
if [ ! -d "visualizer" ]; then
    echo "❌ Error: visualizer directory not found"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found"
    echo "Please install Python 3.6 or higher"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "🐍 Using Python: $PYTHON_CMD"

# Check if pip is available
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "❌ Error: pip not found"
    echo "Please install pip"
    exit 1
fi

# Use pip3 if available, otherwise pip
PIP_CMD="pip3"
if ! command -v pip3 &> /dev/null; then
    PIP_CMD="pip"
fi

echo "📦 Using pip: $PIP_CMD"

# Install dependencies if requirements.txt exists
if [ -f "visualizer/requirements.txt" ]; then
    echo "📚 Installing dependencies..."
    $PIP_CMD install -r visualizer/requirements.txt
    if [ $? -ne 0 ]; then
        echo "⚠️  Warning: Failed to install some dependencies"
        echo "You may need to install Flask manually: $PIP_CMD install Flask"
    fi
else
    echo "⚠️  Requirements file not found, trying to install Flask..."
    $PIP_CMD install Flask
fi

echo ""
echo "🚀 Starting web server..."
echo "📍 Navigate to: http://localhost:5001"
echo "💡 Press Ctrl+C to stop the server"
echo ""

# Change to web directory and start the Flask app
cd visualizer/web

# Start the Flask application
$PYTHON_CMD app.py

echo ""
echo "👋 Visualizer stopped. Thanks for using Agent Workflow Visualizer!"