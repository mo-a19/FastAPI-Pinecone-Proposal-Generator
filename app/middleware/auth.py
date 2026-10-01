"""
Multi-Tenant Auth Middleware

Roles:
- super_admin: Platform admin (env var ADMIN_API_KEY)
- admin: Organization admin (full org access)
- member: Regular user (generate proposals only)

All authenticated requests return user context with org_id for query scoping.
"""
import hashlib
from datetime import datetime
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from app.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def get_master_key() -> str:
    return settings.ADMIN_API_KEY or "dev-key"


async def verify_key(api_key: Optional[str] = Security(api_key_header)) -> dict:
    """
    Verify API key and return user context.
    
    Returns dict with: role, name, permissions, user_id, org_id
    """
    if not api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "API key required")
    
    # Super Admin check (platform-wide)
    if api_key == get_master_key():
        return {
            "role": "super_admin",
            "name": "Super Admin",
            "user_id": None,
            "org_id": None,  # Super admin has no org scope
            "permissions": ["admin", "generate", "read", "write", "training", "manage_org"]
        }
    
    # Try new users collection first
    from app.infra.mongodb.repositories import get_user_repo, UserRole
    user_repo = get_user_repo()
    user = user_repo.get_by_api_key(api_key)
    
    if user:
        role = UserRole(user.get("role", "member"))
        return {
            "role": role.value,
            "name": user.get("name", "User"),
            "user_id": user.get("user_id"),
            "org_id": user.get("org_id"),
            "permissions": role.permissions
        }
    
    # Fallback: Check legacy api_keys collection for backward compatibility
    from app.db import get_db
    db = get_db()
    doc = db.db["api_keys"].find_one({"key_hash": hash_key(api_key)})
    
    if doc:
        if not doc.get("is_active", True):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Key revoked")
        if doc.get("expires_at") and doc["expires_at"] < datetime.utcnow():
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Key expired")
        
        db.db["api_keys"].update_one({"_id": doc["_id"]}, {"$set": {"last_used": datetime.utcnow()}})
        role = doc.get("role", "member")
        perms = ["admin", "generate", "read", "write", "training"] if role == "admin" else ["generate"]
        return {
            "role": role,
            "name": doc.get("name", "User"),
            "user_id": None,
            "org_id": None,  # Legacy keys have no org scope
            "permissions": perms
        }
    
    raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid API key")


async def verify_super_admin(api_key: Optional[str] = Security(api_key_header)) -> dict:
    """Only platform super admin (master key from .env)."""
    if not api_key or api_key != get_master_key():
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Super Admin access required")
    return {"role": "super_admin", "name": "Super Admin", "user_id": None, "org_id": None}


async def verify_admin(api_key: Optional[str] = Security(api_key_header)) -> dict:
    """Admin or Super Admin access."""
    result = await verify_key(api_key)
    if result["role"] not in ("admin", "super_admin"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access required")
    return result


async def verify_user(api_key: Optional[str] = Security(api_key_header)) -> dict:
    """Any valid authenticated user."""
    return await verify_key(api_key)


# Aliases for backward compatibility
verify_api_key = verify_key
verify_admin_key = verify_super_admin
verify_generate_permission = verify_user
verify_training_permission = verify_admin
verify_write_permission = verify_admin
