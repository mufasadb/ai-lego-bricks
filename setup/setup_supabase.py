#!/usr/bin/env python3
"""
Supabase pgvector Setup Script

This script helps you set up pgvector in your Supabase instance for the memory service.
Run this after creating your Supabase project and configuring your .env file.
"""

import os
import sys
from supabase import create_client

# Import credential manager
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from credentials import CredentialManager


def load_environment():
    """Load environment variables and validate Supabase credentials"""
    # Use credential manager with environment loading
    cred_manager = CredentialManager(load_env=True)

    url = cred_manager.get_credential("SUPABASE_URL")
    key = cred_manager.get_credential("SUPABASE_ANON_KEY")

    if not url or not key:
        print("❌ Error: SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env file")
        print(
            "📝 Please copy .env.example to .env and fill in your Supabase credentials"
        )
        return None, None

    return url, key


def check_connection(supabase):
    """Test the connection to Supabase"""
    try:
        # Try to access the memories table (which should exist)
        supabase.table("memories").select("*").limit(1).execute()
        print("✅ Successfully connected to Supabase")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        print("💡 Check your SUPABASE_URL and SUPABASE_ANON_KEY in .env file")
        return False


def check_pgvector_extension(supabase):
    """Check if pgvector extension is installed by testing vector operations"""
    try:
        # Test if we can create a simple vector - this will fail if pgvector isn't installed
        test_vector = "[" + ",".join(["0"] * 384) + "]"

        # Try to call the match_memories function - this requires pgvector
        supabase.rpc(
            "match_memories",
            {"query_embedding": test_vector, "match_threshold": 0.1, "match_count": 1},
        ).execute()

        print("✅ pgvector extension is working (vector operations successful)")
        return True
    except Exception as e:
        if "vector" in str(e).lower():
            print("⚠️  pgvector extension not found or not working")
        else:
            print(f"❌ Error checking pgvector extension: {e}")
        return False


def check_memories_table(supabase):
    """Check if memories table exists with proper schema"""
    try:
        # Check if table exists
        supabase.table("memories").select("*").limit(1).execute()
        print("✅ memories table exists")

        # Try to insert a test record to verify the schema
        test_data = {"content": "Schema verification test", "metadata": {"test": True}}

        # Insert test record
        insert_result = supabase.table("memories").insert(test_data).execute()
        if insert_result.data:
            test_id = insert_result.data[0]["id"]
            print("✅ Table schema is correct (can insert records)")

            # Clean up test record
            supabase.table("memories").delete().eq("id", test_id).execute()
            return True
        else:
            print("⚠️  Could not insert test record")
            return False

    except Exception as e:
        print(f"⚠️  memories table does not exist or is not accessible: {e}")
        return False


def check_match_function(supabase):
    """Check if match_memories function exists"""
    try:
        # Create a simple test vector (all zeros)
        test_vector = "[" + ",".join(["0"] * 384) + "]"

        # Try to call the match_memories function
        supabase.rpc(
            "match_memories",
            {"query_embedding": test_vector, "match_threshold": 0.1, "match_count": 1},
        ).execute()

        print("✅ match_memories function exists and is callable")
        return True
    except Exception as e:
        print(f"❌ Error checking match_memories function: {e}")
        return False


def test_vector_operations(supabase):
    """Test basic vector operations"""
    try:
        from sentence_transformers import SentenceTransformer

        # Generate a test embedding
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        test_embedding = encoder.encode("This is a test sentence for vector operations")
        embedding_str = "[" + ",".join(map(str, test_embedding.tolist())) + "]"

        # Test inserting a memory
        test_data = {
            "content": "Test memory for pgvector setup verification",
            "embedding": embedding_str,
            "metadata": {"test": True, "setup_verification": True},
        }

        insert_result = supabase.table("memories").insert(test_data).execute()
        if insert_result.data:
            test_id = insert_result.data[0]["id"]
            print("✅ Successfully inserted test memory")

            # Test vector search
            search_result = supabase.rpc(
                "match_memories",
                {
                    "query_embedding": embedding_str,
                    "match_threshold": 0.1,
                    "match_count": 1,
                },
            ).execute()

            if search_result.data and len(search_result.data) > 0:
                print("✅ Vector search is working!")

                # Clean up test data
                supabase.table("memories").delete().eq("id", test_id).execute()
                print("✅ Cleaned up test data")
                return True
            else:
                print("⚠️  Vector search returned no results")
                # Clean up anyway
                supabase.table("memories").delete().eq("id", test_id).execute()
                return False
        else:
            print("❌ Failed to insert test memory")
            return False

    except ImportError:
        print("⚠️  sentence-transformers not installed, skipping vector test")
        print("   Run: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"❌ Error testing vector operations: {e}")
        return False


def main():
    """Main setup verification function"""
    print("🚀 Supabase pgvector Setup Verification")
    print("=" * 50)

    # Load environment
    url, key = load_environment()
    if not url or not key:
        return False

    # Create Supabase client
    try:
        supabase = create_client(url, key)
    except Exception as e:
        print(f"❌ Failed to create Supabase client: {e}")
        return False

    # Run checks
    checks = [
        ("Connection", lambda: check_connection(supabase)),
        ("pgvector Extension", lambda: check_pgvector_extension(supabase)),
        ("memories Table", lambda: check_memories_table(supabase)),
        ("match_memories Function", lambda: check_match_function(supabase)),
        ("Vector Operations", lambda: test_vector_operations(supabase)),
    ]

    results = []
    for name, check_func in checks:
        print(f"\n🔍 Checking {name}...")
        result = check_func()
        results.append((name, result))

    # Summary
    print("\n📊 Setup Summary:")
    print("=" * 50)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All checks passed! Your Supabase instance is ready for pgvector.")
        print("💡 You can now use the SupabaseMemoryService in your application.")
    else:
        print("\n⚠️  Some checks failed. Please review the setup:")
        print("\n📝 To fix issues:")
        print("1. Open your Supabase project dashboard")
        print("2. Go to the SQL Editor")
        print("3. Run the contents of setup_supabase_pgvector.sql")
        print("4. Run this script again to verify")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
