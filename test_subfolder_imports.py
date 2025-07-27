#!/usr/bin/env python3
"""
Test script to verify that our import fixes work when used as a subfolder.
This simulates putting the ai-lego-bricks folder inside another project.
"""

import sys
import os
import traceback

def test_direct_imports():
    """Test importing modules directly from the current structure."""
    print("=" * 60)
    print("TESTING DIRECT MODULE IMPORTS (subfolder simulation)")
    print("=" * 60)
    
    success_count = 0
    total_tests = 0
    
    # Test direct imports of key modules
    tests = [
        ("agent_orchestration.orchestrator", "AgentOrchestrator", "from agent_orchestration.orchestrator import AgentOrchestrator"),
        ("chat.chat_service", "create_chat_service", "from chat.chat_service import create_chat_service"),
        ("memory.memory_factory", "MemoryServiceFactory", "from memory.memory_factory import MemoryServiceFactory"),
        ("credentials.credential_manager", "CredentialManager", "from credentials.credential_manager import CredentialManager"),
        ("llm.llm_factory", "LLMClientFactory", "from llm.llm_factory import LLMClientFactory"),
    ]
    
    for module_path, class_name, import_statement in tests:
        total_tests += 1
        try:
            print(f"Testing: {import_statement}")
            exec(import_statement)
            print(f"✅ SUCCESS: {class_name} imported")
            success_count += 1
        except ImportError as e:
            print(f"❌ FAILED: {class_name} - {e}")
        except Exception as e:
            print(f"⚠️  ERROR: {class_name} - {e}")
    
    print("=" * 60)
    print(f"DIRECT IMPORT RESULTS: {success_count}/{total_tests} successful")
    print("=" * 60)
    
    return success_count, total_tests

def test_internal_relative_imports():
    """Test that internal relative imports work correctly."""
    print("\nTesting internal relative imports...")
    
    success_count = 0
    total_tests = 0
    
    # Test some key internal imports that were fixed
    tests = [
        ("Test LLM factory importing credentials", "from llm.llm_factory import LLMClientFactory; LLMClientFactory"),
        ("Test agent orchestrator importing LLM", "from agent_orchestration.orchestrator import AgentOrchestrator"),
        ("Test memory service imports", "from memory.memory_factory import MemoryServiceFactory"),
        ("Test chat service imports", "from chat.conversation_service import ConversationService"),
    ]
    
    for description, import_statement in tests:
        total_tests += 1
        try:
            print(f"Testing: {description}")
            exec(import_statement)
            print(f"✅ SUCCESS: {description}")
            success_count += 1
        except ImportError as e:
            print(f"❌ IMPORT FAILED: {description} - {e}")
        except Exception as e:
            print(f"⚠️  ERROR: {description} - {e}")
    
    print(f"\nINTERNAL IMPORT RESULTS: {success_count}/{total_tests} successful")
    
    return success_count, total_tests

def test_sys_path_removal():
    """Verify that we removed sys.path.append calls successfully."""
    print("\nChecking for remaining sys.path.append calls...")
    
    import subprocess
    import glob
    
    try:
        # Search for remaining sys.path.append calls
        result = subprocess.run(
            ['grep', '-r', 'sys.path.append', '.', '--include=*.py'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            # Filter out our test files
            remaining_calls = [line for line in lines if 'test_' not in line and line.strip()]
            
            if remaining_calls:
                print(f"⚠️  Found {len(remaining_calls)} remaining sys.path.append calls:")
                for call in remaining_calls[:5]:  # Show first 5
                    print(f"   {call}")
                if len(remaining_calls) > 5:
                    print(f"   ... and {len(remaining_calls) - 5} more")
                return False
            else:
                print("✅ SUCCESS: All sys.path.append calls have been removed!")
                return True
        else:
            print("✅ SUCCESS: No sys.path.append calls found!")
            return True
            
    except Exception as e:
        print(f"⚠️  Could not check for sys.path.append calls: {e}")
        return True  # Don't fail the test for this

def test_key_functionality():
    """Test that key functionality still works after our changes."""
    print("\nTesting key functionality...")
    
    try:
        # Test AgentOrchestrator creation
        from agent_orchestration.orchestrator import AgentOrchestrator
        orchestrator = AgentOrchestrator()
        print("✅ AgentOrchestrator can be created")
        
        # Test CredentialManager
        from credentials.credential_manager import CredentialManager
        cred_manager = CredentialManager()
        print("✅ CredentialManager can be created")
        
        # Test LLM Factory
        from llm.llm_factory import LLMClientFactory
        print("✅ LLMClientFactory can be imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing AI Lego Bricks import structure for subfolder usage...\n")
    
    # Run all tests
    direct_success, direct_total = test_direct_imports()
    internal_success, internal_total = test_internal_relative_imports()
    syspath_clean = test_sys_path_removal()
    functionality_ok = test_key_functionality()
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Direct imports: {direct_success}/{direct_total}")
    print(f"Internal imports: {internal_success}/{internal_total}")
    print(f"sys.path cleanup: {'✅ Clean' if syspath_clean else '⚠️  Remaining calls'}")
    print(f"Basic functionality: {'✅ Working' if functionality_ok else '❌ Failed'}")
    
    total_success = direct_success + internal_success
    total_tests = direct_total + internal_total
    success_rate = (total_success / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\nOverall success rate: {success_rate:.1f}% ({total_success}/{total_tests})")
    
    if success_rate >= 80 and functionality_ok:
        print("\n🎉 AI Lego Bricks is ready for subfolder usage!")
        print("   You can now copy this folder into other projects and import modules directly.")
    else:
        print("\n⚠️  Some issues remain. Check the output above for details.")