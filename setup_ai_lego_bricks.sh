#!/bin/bash
# AI Lego Bricks Setup Script
# Automates the installation process for new users

set -e  # Exit on any error

echo "🧱 AI Lego Bricks Setup Script"
echo "================================"

# Check Python version
echo "📋 Checking Python version..."
python3 --version || {
    echo "❌ Python 3 is required but not found. Please install Python 3.8+"
    exit 1
}

# Upgrade pip to prevent editable install issues
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Parse command line argument for install type
INSTALL_TYPE=${1:-minimal}

case $INSTALL_TYPE in
    minimal)
        echo "📦 Installing minimal dependencies..."
        pip install -e ".[minimal]"
        ;;
    ollama)
        echo "📦 Installing for Ollama (local models)..."
        pip install -e ".[ollama]"
        ;;
    openai)
        echo "📦 Installing for OpenAI..."
        pip install -e ".[openai]"
        ;;
    gemini)
        echo "📦 Installing for Google Gemini..."
        pip install -e ".[gemini]"
        ;;
    llm)
        echo "📦 Installing all LLM providers..."
        pip install -e ".[llm]"
        ;;
    all)
        echo "📦 Installing all features (this may take a while)..."
        pip install -e ".[all]"
        ;;
    *)
        echo "❌ Unknown install type: $INSTALL_TYPE"
        echo "Available options: minimal, ollama, openai, gemini, llm, all"
        exit 1
        ;;
esac

# Setup environment file
echo "🔧 Setting up environment configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file from template"
else
    echo "ℹ️  .env file already exists, skipping"
fi

# Success message with next steps
echo ""
echo "🎉 AI Lego Bricks installation complete!"
echo ""
echo "📝 Next steps:"
case $INSTALL_TYPE in
    ollama)
        echo "   1. Start Ollama: 'ollama serve'"
        echo "   2. Pull a model: 'ollama pull llama3.1:8b'"
        echo "   3. Set OLLAMA_URL in .env: 'http://localhost:11434'"
        echo "   4. Test: 'ailego run agent_orchestration/examples/basic_chat_agent.json'"
        ;;
    openai)
        echo "   1. Add your OpenAI API key to .env: OPENAI_API_KEY=your_key"
        echo "   2. Test: 'ailego run agent_orchestration/examples/basic_chat_agent.json'"
        ;;
    gemini)
        echo "   1. Add your Google API key to .env: GOOGLE_API_KEY=your_key"
        echo "   2. Test: 'ailego run agent_orchestration/examples/basic_chat_agent.json'"
        ;;
    *)
        echo "   1. Edit .env with your API keys (see .env.example for guidance)"
        echo "   2. Test: 'ailego verify'"
        echo "   3. Run: 'ailego run agent_orchestration/examples/basic_chat_agent.json'"
        ;;
esac
echo ""
echo "📚 Documentation: https://github.com/callmebeachy/ai-lego-bricks#readme"
echo "🆘 Issues: https://github.com/callmebeachy/ai-lego-bricks/issues"