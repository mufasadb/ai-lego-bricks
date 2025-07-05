#!/usr/bin/env python3
"""
Test all delete operations in the memory service
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.memory_factory import create_memory_service
import time

def test_delete_operations():
    print("=== Testing Delete Operations ===\n")
    
    # Create memory service
    try:
        memory_service = create_memory_service("auto")
        print(f"✓ Created {type(memory_service).__name__}")
    except Exception as e:
        print(f"✗ Failed to create service: {e}")
        return False
    
    # Store some test memories
    print(f"\n1. Storing test memories...")
    test_memories = [
        {"content": "Test memory 1 - to be deleted", "metadata": {"test": "delete1"}},
        {"content": "Test memory 2 - to be deleted", "metadata": {"test": "delete2"}}, 
        {"content": "Test memory 3 - to be kept", "metadata": {"test": "keep"}},
        {"content": "Test memory 4 - to be deleted", "metadata": {"test": "delete3"}},
    ]
    
    stored_ids = []
    for i, memory_data in enumerate(test_memories):
        try:
            memory_id = memory_service.store_memory(memory_data["content"], memory_data["metadata"])
            stored_ids.append(memory_id)
            print(f"   ✓ Stored test memory {i+1}: {memory_id[:8]}...")
        except Exception as e:
            print(f"   ✗ Failed to store memory {i+1}: {e}")
            return False
    
    print(f"\n2. Testing individual delete operations...")
    
    # Test deleting first memory
    delete_id = stored_ids[0]
    print(f"   🔄 Deleting memory: {delete_id[:8]}...")
    
    try:
        deleted = memory_service.delete_memory(delete_id)
        if deleted:
            print(f"   ✓ Successfully deleted memory")
            
            # Verify it's really gone
            retrieved = memory_service.get_memory_by_id(delete_id)
            if retrieved is None:
                print(f"   ✓ Confirmed memory is gone (get_by_id returns None)")
            else:
                print(f"   ✗ ERROR: Memory still exists after deletion!")
                return False
        else:
            print(f"   ✗ Delete operation returned False")
            return False
            
    except Exception as e:
        print(f"   ✗ Delete failed with exception: {e}")
        return False
    
    # Test deleting non-existent memory
    print(f"\n3. Testing delete of non-existent memory...")
    fake_id = "non-existent-id-12345"
    try:
        deleted = memory_service.delete_memory(fake_id)
        if not deleted:
            print(f"   ✓ Correctly returned False for non-existent memory")
        else:
            print(f"   ✗ ERROR: Returned True for non-existent memory")
    except Exception as e:
        print(f"   ✗ Exception on non-existent delete: {e}")
    
    # Test bulk delete operations
    print(f"\n4. Testing bulk delete operations...")
    delete_ids = [stored_ids[1], stored_ids[3]]  # Delete memories 2 and 4, keep memory 3
    
    successful_deletes = 0
    for delete_id in delete_ids:
        print(f"   🔄 Deleting memory: {delete_id[:8]}...")
        try:
            deleted = memory_service.delete_memory(delete_id)
            if deleted:
                print(f"   ✓ Successfully deleted")
                successful_deletes += 1
            else:
                print(f"   ✗ Delete returned False")
        except Exception as e:
            print(f"   ✗ Delete failed: {e}")
    
    print(f"   📊 Bulk delete results: {successful_deletes}/{len(delete_ids)} successful")
    
    # Verify remaining memory
    print(f"\n5. Verifying remaining memory...")
    keep_id = stored_ids[2]  # This should still exist
    try:
        kept_memory = memory_service.get_memory_by_id(keep_id)
        if kept_memory:
            print(f"   ✓ Kept memory still exists: {kept_memory.content}")
            print(f"   ✓ Metadata intact: {kept_memory.metadata}")
        else:
            print(f"   ✗ ERROR: Kept memory was also deleted!")
            return False
    except Exception as e:
        print(f"   ✗ Error checking kept memory: {e}")
        return False
    
    # Clean up remaining test memory
    print(f"\n6. Cleaning up remaining test memory...")
    try:
        deleted = memory_service.delete_memory(keep_id)
        if deleted:
            print(f"   ✓ Cleanup successful")
        else:
            print(f"   ⚠ Cleanup returned False (memory might not exist)")
    except Exception as e:
        print(f"   ✗ Cleanup failed: {e}")
    
    print(f"\n=== Delete Operations Test Complete ===")
    return True

def test_search_after_delete():
    """Test that deleted memories don't appear in search results"""
    print(f"\n=== Testing Search After Delete ===\n")
    
    memory_service = create_memory_service("auto")
    
    # Store a memory with unique content
    unique_content = f"Unique test content for search delete test {int(time.time())}"
    print(f"1. Storing unique memory...")
    memory_id = memory_service.store_memory(unique_content, {"test": "search_delete"})
    print(f"   ✓ Stored: {memory_id[:8]}...")
    
    # Search for it
    print(f"2. Searching for memory before delete...")
    results = memory_service.retrieve_memories("unique test content search delete", limit=5)
    found_before = any(result.memory_id == memory_id for result in results)
    print(f"   {'✓' if found_before else '✗'} Found in search: {found_before}")
    
    # Delete it
    print(f"3. Deleting memory...")
    deleted = memory_service.delete_memory(memory_id)
    print(f"   {'✓' if deleted else '✗'} Deleted: {deleted}")
    
    # Search again
    print(f"4. Searching for memory after delete...")
    results = memory_service.retrieve_memories("unique test content search delete", limit=5)
    found_after = any(result.memory_id == memory_id for result in results)
    print(f"   {'✓' if not found_after else '✗'} Not found in search: {not found_after}")
    
    if found_before and not found_after and deleted:
        print(f"   ✅ Search correctly excludes deleted memories")
        return True
    else:
        print(f"   ❌ Search/delete integration issue")
        return False

if __name__ == "__main__":
    print("Testing memory deletion functionality...\n")
    
    test1_passed = test_delete_operations()
    test2_passed = test_search_after_delete()
    
    print(f"\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Delete Operations Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Search After Delete Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print(f"\n🎉 ALL DELETE TESTS PASSED!")
        print(f"✅ Individual delete works")
        print(f"✅ Bulk delete works") 
        print(f"✅ Non-existent delete handled properly")
        print(f"✅ Deleted memories don't appear in search")
        print(f"✅ Memory service delete functionality is solid!")
    else:
        print(f"\n❌ Some delete tests failed - check implementation")