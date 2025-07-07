from typing import Dict, Any, Optional
import os
from .memory_service import MemoryService
try:
    from ..credentials import CredentialManager, default_credential_manager
except ImportError:
    # Fallback for when running as standalone
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from credentials import CredentialManager, default_credential_manager

class MemoryServiceFactory:
    """Factory for creating memory service instances"""
    
    @staticmethod
    def create_memory_service(service_type: str = "auto", credential_manager: Optional[CredentialManager] = None, **kwargs) -> MemoryService:
        """
        Create a memory service instance
        
        Args:
            service_type: Type of service ("supabase", "neo4j", or "auto")
            credential_manager: Optional credential manager for explicit credential handling
            **kwargs: Additional parameters for the specific service
            
        Returns:
            MemoryService instance
        """
        service_type = service_type.lower()
        
        if service_type == "auto":
            service_type = MemoryServiceFactory._detect_available_service(credential_manager)
        
        if service_type == "supabase":
            return MemoryServiceFactory._create_supabase_service(credential_manager, **kwargs)
        elif service_type == "neo4j":
            return MemoryServiceFactory._create_neo4j_service(credential_manager, **kwargs)
        else:
            raise ValueError(f"Unsupported memory service type: {service_type}")
    
    @staticmethod
    def _detect_available_service(credential_manager: Optional[CredentialManager] = None) -> str:
        """Detect which memory service is available based on environment variables"""
        cred_manager = credential_manager or default_credential_manager
        
        # Check Neo4j first (simpler setup)
        neo4j_uri = cred_manager.get_credential("NEO4J_URI") or cred_manager.get_credential("NEO4J_URL")
        if neo4j_uri:
            return "neo4j"
        
        # Check Supabase
        supabase_url = cred_manager.get_credential("SUPABASE_URL")
        supabase_key = cred_manager.get_credential("SUPABASE_ANON_KEY")
        if supabase_url and supabase_key:
            return "supabase"
        
        raise RuntimeError(
            "No memory service found. Set NEO4J_URI or SUPABASE_URL/SUPABASE_ANON_KEY"
        )
    
    @staticmethod
    def _create_supabase_service(credential_manager: Optional[CredentialManager] = None, **kwargs) -> MemoryService:
        """Create Supabase memory service instance"""
        try:
            from .supabase_memory_service import SupabaseMemoryService
            return SupabaseMemoryService(credential_manager=credential_manager, **kwargs)
        except ImportError as e:
            raise ImportError(f"Failed to import Supabase dependencies: {e}")
    
    @staticmethod
    def _create_neo4j_service(credential_manager: Optional[CredentialManager] = None, **kwargs) -> MemoryService:
        """Create Neo4j memory service instance"""
        try:
            from .neo4j_memory_service import Neo4jMemoryService
            return Neo4jMemoryService(credential_manager=credential_manager, **kwargs)
        except ImportError as e:
            raise ImportError(f"Failed to import Neo4j dependencies: {e}")
    
    @staticmethod
    def get_available_services() -> Dict[str, bool]:
        """Check which memory services are available"""
        return get_available_services()

# Convenience functions
def create_memory_service(service_type: str = "auto", credential_manager: Optional[CredentialManager] = None, **kwargs) -> MemoryService:
    """Create a memory service instance (convenience function)"""
    return MemoryServiceFactory.create_memory_service(service_type, credential_manager, **kwargs)

def get_available_services() -> Dict[str, bool]:
    """Check which memory services are available"""
    services = {"supabase": False, "neo4j": False}
    
    # Check Neo4j
    if os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL"):
        try:
            from .neo4j_memory_service import Neo4jMemoryService
            services["neo4j"] = True
        except ImportError:
            pass
    
    # Check Supabase
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_ANON_KEY"):
        try:
            from .supabase_memory_service import SupabaseMemoryService
            services["supabase"] = True
        except ImportError:
            pass
    
    return services