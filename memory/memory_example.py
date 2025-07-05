"""
Example usage of the memory service
"""

from .memory_factory import create_memory_service, get_available_services
from .memory_service import Memory
import json

def main():
    print("=== Memory Service Example ===\n")
    
    # Check available services
    print("1. Checking available services:")
    available = get_available_services()
    for service, is_available in available.items():
        status = "✓" if is_available else "✗"
        print(f"   {status} {service.capitalize()}")
    
    # If no services available, show what's needed
    if not any(available.values()):
        print("\n❌ No memory services available!")
        print("Please set environment variables:")
        print("  For Supabase: SUPABASE_URL and SUPABASE_ANON_KEY")
        print("  For Neo4j: NEO4J_PASSWORD (and optionally NEO4J_URI, NEO4J_USERNAME)")
        return
    
    print("\n2. Creating memory service (auto-detect):")
    try:
        memory_service = create_memory_service("auto")
        print(f"   ✓ Created {type(memory_service).__name__}")
    except Exception as e:
        print(f"   ✗ Failed to create service: {e}")
        return
    
    print("\n3. Storing memories:")
    
    # Store some sample memories
    new_memories_to_store = [
        {
            "content": "Client called about the db optimizations",
            "metadata": {"type": "call", "client": "Acme Corp", "topic": "optimization"}
        }
	]
    memories_to_store = [
        {
            "content": "Meeting with client about project requirements. They need a dashboard for analytics.",
            "metadata": {"type": "meeting", "client": "Acme Corp", "topic": "requirements"}
        },
        {
            "content": "Client called about the db optimizations",
            "metadata": {"type": "call", "client": "Acme Corp", "topic": "optimization"}
        },
        {
            "content": "Implemented user authentication using JWT tokens. Need to add refresh token logic.",
            "metadata": {"type": "development", "component": "auth", "status": "in_progress"}
        },
        {
            "content": "Database optimization reduced query time by 40%. Used indexing on user_id column.",
            "metadata": {"type": "optimization", "component": "database", "improvement": "40%"}
        },
        {
            "content": "Bug reported: users can't upload files larger than 10MB. Need to check server config.",
            "metadata": {"type": "bug", "component": "file_upload", "priority": "high"}
        }
    ]
    
    stored_ids = []
    for i, memory_data in enumerate(new_memories_to_store):
        try:
            memory_id = memory_service.store_memory(
                memory_data["content"], 
                memory_data["metadata"]
            )
            stored_ids.append(memory_id)
            print(f"   ✓ Stored memory {i+1}: {memory_id[:8]}...")
        except Exception as e:
            print(f"   ✗ Failed to store memory {i+1}: {e}")
    
    print(f"\n4. Retrieving memories by similarity:")
    
    # Test queries
    test_queries = [
        "authentication problems",
        "database performance",
        "client meeting notes",
        "file upload issues"
    ]
    
    for query in test_queries:
        try:
            results = memory_service.retrieve_memories(query, limit=2)
            print(f"\n   Query: '{query}'")
            print(f"   Found {len(results)} memories:")
            
            for j, memory in enumerate(results):
                print(f"     {j+1}. {memory.content[:80]}...")
                if memory.metadata:
                    print(f"        Metadata: {json.dumps(memory.metadata, indent=8)}")
                    
        except Exception as e:
            print(f"   ✗ Query '{query}' failed: {e}")
    
    print(f"\n5. Getting specific memory by ID:")
    if stored_ids:
        try:
            memory = memory_service.get_memory_by_id(stored_ids[0])
            if memory:
                print(f"   ✓ Retrieved: {memory.content[:80]}...")
                print(f"   Timestamp: {memory.timestamp}")
            else:
                print("   ✗ Memory not found")
        except Exception as e:
            print(f"   ✗ Failed to retrieve memory: {e}")
    
    print(f"\n6. Updating a memory:")
    if stored_ids:
        try:
            updated = memory_service.update_memory(
                stored_ids[0],
                "Updated: Meeting with client about project requirements. Dashboard approved, starting development next week.",
                {"type": "meeting", "client": "Acme Corp", "topic": "requirements", "status": "approved"}
            )
            if updated:
                print("   ✓ Memory updated successfully")
            else:
                print("   ✗ Failed to update memory")
        except Exception as e:
            print(f"   ✗ Update failed: {e}")
    
    print(f"\n7. Deleting a memory:")
    if len(stored_ids) > 1:
        try:
            deleted = memory_service.delete_memory(stored_ids[-1])
            if deleted:
                print("   ✓ Memory deleted successfully")
            else:
                print("   ✗ Failed to delete memory")
        except Exception as e:
            print(f"   ✗ Delete failed: {e}")
    
    print(f"\n=== Example Complete ===")

if __name__ == "__main__":
    main()