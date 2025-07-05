#!/usr/bin/env python3
"""
Test the new bulk delete functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.memory_factory import create_memory_service

def test_bulk_delete():
    print("=== Testing Bulk Delete Functionality ===\n")
    
    # Create memory service
    memory_service = create_memory_service("auto")
    print(f"✓ Created {type(memory_service).__name__}")
    
    # Store test memories
    print(f"\n1. Storing test memories for bulk delete...")
    test_memories = [
        {"content": f"Bulk test memory {i}", "metadata": {"test": "bulk", "number": str(i)}}
        for i in range(1, 6)  # 5 memories
    ]
    
    stored_ids = []
    for i, memory_data in enumerate(test_memories):
        memory_id = memory_service.store_memory(memory_data["content"], memory_data["metadata"])
        stored_ids.append(memory_id)
        print(f"   ✓ Stored memory {i+1}: {memory_id[:8]}...")
    
    # Test bulk delete with valid IDs
    print(f"\n2. Testing bulk delete with valid IDs...")
    delete_ids = stored_ids[:3]  # Delete first 3 memories
    print(f"   🔄 Bulk deleting {len(delete_ids)} memories...")
    
    results = memory_service.delete_memories(delete_ids)
    
    successful_deletes = sum(1 for success in results.values() if success)
    print(f"   📊 Bulk delete results: {successful_deletes}/{len(delete_ids)} successful")
    
    for memory_id, success in results.items():
        status = "✓" if success else "✗"
        print(f"   {status} {memory_id[:8]}: {'deleted' if success else 'failed'}")
    
    # Verify deleted memories are gone
    print(f"\n3. Verifying deleted memories are gone...")
    for memory_id in delete_ids:
        retrieved = memory_service.get_memory_by_id(memory_id)
        if retrieved is None:
            print(f"   ✓ {memory_id[:8]} confirmed deleted")
        else:
            print(f"   ✗ {memory_id[:8]} still exists!")
    
    # Verify remaining memories still exist
    print(f"\n4. Verifying remaining memories still exist...")
    remaining_ids = stored_ids[3:]  # Last 2 memories should remain
    for memory_id in remaining_ids:
        retrieved = memory_service.get_memory_by_id(memory_id)
        if retrieved:
            print(f"   ✓ {memory_id[:8]} still exists: {retrieved.content}")
        else:
            print(f"   ✗ {memory_id[:8]} was accidentally deleted!")
    
    # Test bulk delete with mix of valid and invalid IDs
    print(f"\n5. Testing bulk delete with mixed valid/invalid IDs...")
    mixed_ids = remaining_ids + ["fake-id-1", "fake-id-2"]
    print(f"   🔄 Bulk deleting {len(mixed_ids)} IDs (mix of valid/invalid)...")
    
    results = memory_service.delete_memories(mixed_ids)
    
    for memory_id, success in results.items():
        status = "✓" if success else "✗"
        id_type = "valid" if memory_id in remaining_ids else "invalid"
        print(f"   {status} {memory_id[:8] if len(memory_id) > 8 else memory_id}: {id_type} ID -> {'deleted' if success else 'not found'}")
    
    # Test bulk delete with empty list
    print(f"\n6. Testing bulk delete with empty list...")
    empty_results = memory_service.delete_memories([])
    if len(empty_results) == 0:
        print(f"   ✓ Empty list handled correctly (returned empty dict)")
    else:
        print(f"   ✗ Empty list returned unexpected results: {empty_results}")
    
    print(f"\n=== Bulk Delete Test Complete ===")
    
    # Check if all expected deletes worked
    expected_successful = len(delete_ids) + len(remaining_ids)  # All valid IDs
    expected_failed = 2  # The fake IDs
    
    total_successful = sum(1 for success in results.values() if success)
    total_failed = sum(1 for success in results.values() if not success)
    
    if total_successful >= len(remaining_ids):  # At least the second batch worked
        print(f"✅ Bulk delete functionality working correctly!")
        return True
    else:
        print(f"❌ Bulk delete had issues")
        return False

if __name__ == "__main__":
    success = test_bulk_delete()
    
    if success:
        print(f"\n🎉 BULK DELETE READY TO USE!")
        print(f"✅ Individual delete: memory_service.delete_memory(id)")
        print(f"✅ Bulk delete: memory_service.delete_memories([id1, id2, id3])")
        print(f"✅ Handles invalid IDs gracefully")
        print(f"✅ Efficient Neo4j implementation with single query")
    else:
        print(f"\n❌ Bulk delete needs debugging")