#!/usr/bin/env python3
"""
Test script to verify that our import fixes work correctly.
This simulates how the project would be imported when used as a subfolder.
"""

import sys
import traceback

def test_import(module_name, description):
    """Test importing a module and report the result."""
    try:
        print(f"Testing {description}...")
        exec(f"import {module_name}")
        print(f"✅ SUCCESS: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ FAILED: {module_name} - {e}")
        return False
    except Exception as e:
        print(f"⚠️  ERROR: {module_name} - {e}")
        return False

def test_relative_imports():
    """Test that relative imports work correctly."""
    print("=" * 60)
    print("TESTING AI LEGO BRICKS IMPORT STRUCTURE")
    print("=" * 60)
    
    # Test core package import
    success_count = 0
    total_tests = 0
    
    tests = [
        ("ai_lego_bricks", "Root package"),
        ("ai_lego_bricks.agent_orchestration", "Agent Orchestration"),
        ("ai_lego_bricks.chat", "Chat Services"), 
        ("ai_lego_bricks.llm", "LLM Services"),
        ("ai_lego_bricks.memory", "Memory Services"),
        ("ai_lego_bricks.credentials", "Credential Management"),
        ("ai_lego_bricks.chunking", "Text Chunking"),
        ("ai_lego_bricks.pdf_to_text", "PDF Processing"),
        ("ai_lego_bricks.tts", "Text-to-Speech"),
        ("ai_lego_bricks.stt", "Speech-to-Text"),
        ("ai_lego_bricks.tools", "Tools & MCP"),
        ("ai_lego_bricks.visualizer", "Workflow Visualizer"),
    ]
    
    for module_name, description in tests:
        total_tests += 1
        if test_import(module_name, description):
            success_count += 1
    
    print("=" * 60)
    print(f"RESULTS: {success_count}/{total_tests} imports successful")
    print("=" * 60)
    
    if success_count == total_tests:
        print("🎉 ALL IMPORTS WORKING! The subfolder structure is ready.")
    else:
        print("⚠️  Some imports failed. Check the error messages above.")
    
    return success_count == total_tests

def test_specific_functionality():
    """Test specific functionality that depends on proper imports."""
    print("\nTesting specific functionality...")
    
    try:
        # Test that we can import the root package
        import ai_lego_bricks
        print(f"✅ Root package version: {ai_lego_bricks.__version__}")
        
        # Test that key components are available
        if hasattr(ai_lego_bricks, 'AgentOrchestrator'):
            print("✅ AgentOrchestrator available")
        else:
            print("❌ AgentOrchestrator not available")
            
        if hasattr(ai_lego_bricks, 'create_chat_service'):
            print("✅ create_chat_service available")
        else:
            print("❌ create_chat_service not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Add the parent directory to path to simulate subfolder usage
    import os
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Run tests
    imports_ok = test_relative_imports()
    functionality_ok = test_specific_functionality()
    
    if imports_ok and functionality_ok:
        print("\n🎉 ALL TESTS PASSED! AI Lego Bricks is ready for subfolder usage.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. See output above for details.")
        sys.exit(1)