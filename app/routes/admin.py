"""
Simplified Admin Routes - 3 Roles: Super Admin / Admin / User

Super Admin (master .env key): Create/manage keys
Admin: Full feature access
User: Generate only
"""
import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, EmailStr
from app.middleware.auth import hash_key, verify_super_admin, verify_admin, verify_key
from app.db import get_db

router = APIRouter(prefix="/api/admin", tags=["Admin"])
logger = logging.getLogger(__name__)

class CreateKeyRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: Optional[EmailStr] = None
    role: str = Field(..., pattern="^(admin|user)$")  # Only admin or user
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)

class KeyResponse(BaseModel):
    id: str
    name: str
    role: str
    email: Optional[str]
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool

# ============ KEY MANAGEMENT ============

@router.post("/keys", dependencies=[Depends(verify_super_admin)])
async def create_key(req: CreateKeyRequest):
    """Super Admin only: Create new API key"""
    db = get_db()
    raw_key = f"pk_{secrets.token_urlsafe(32)}"
    
    doc = {
        "key_hash": hash_key(raw_key),
        "name": req.name,
        "email": req.email,
        "role": req.role,  # "admin" or "user"
        "created_at": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(days=req.expires_in_days) if req.expires_in_days else None,
        "is_active": True
    }
    db.db["api_keys"].insert_one(doc)
    _log_activity(db, "key_created", req.name, {"role": req.role})
    
    return {"success": True, "api_key": raw_key, "message": f"{req.role.title()} key created - save it now, shown only once!"}

@router.get("/keys", dependencies=[Depends(verify_super_admin)])
async def list_keys():
    """Super Admin only: List all keys"""
    db = get_db()
    keys = list(db.db["api_keys"].find({}, {"key_hash": 0}).sort("created_at", -1))
    return {"keys": [_format_key(k) for k in keys], "total": len(keys)}

@router.delete("/keys/{key_id}", dependencies=[Depends(verify_super_admin)])
async def revoke_key(key_id: str):
    """Super Admin only: Revoke key"""
    from bson import ObjectId
    db = get_db()
    result = db.db["api_keys"].update_one({"_id": ObjectId(key_id)}, {"$set": {"is_active": False}})
    if result.modified_count == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Key not found")
    _log_activity(db, "key_revoked", "Super Admin", {"key_id": key_id})
    return {"success": True}

# ============ VERIFICATION ============

@router.get("/verify")
async def verify_current_key(info: dict = Depends(verify_key)):
    """Verify key and return role/permissions"""
    return {"valid": True, **info}

@router.get("/verify-super-admin")
async def verify_super_admin_key(info: dict = Depends(verify_super_admin)):
    """Check if key is Super Admin"""
    return {"valid": True, "is_super_admin": True}

# ============ STATS & ACTIVITY ============

@router.get("/stats", dependencies=[Depends(verify_super_admin)])
async def get_stats():
    """Super Admin only: Dashboard stats"""
    db = get_db()
    now = datetime.utcnow()
    keys = db.db["api_keys"]
    
    return {
        "total_keys": keys.count_documents({}),
        "active_keys": keys.count_documents({"is_active": True}),
        "admin_keys": keys.count_documents({"role": "admin", "is_active": True}),
        "user_keys": keys.count_documents({"role": "user", "is_active": True}),
        "expiring_soon": keys.count_documents({"expires_at": {"$lte": now + timedelta(days=7), "$gt": now}}),
        "recent_activity": _get_recent_activity(db, 5)
    }

@router.get("/activity", dependencies=[Depends(verify_super_admin)])
async def get_activity(limit: int = 50):
    """Super Admin only: Activity log"""
    db = get_db()
    return {"activities": _get_recent_activity(db, limit)}

# ============ HELPERS ============

def _format_key(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "name": doc.get("name", "Unknown"),
        "role": doc.get("role", "user"),
        "email": doc.get("email"),
        "created_at": doc.get("created_at"),
        "expires_at": doc.get("expires_at"),
        "is_active": doc.get("is_active", True),
        "last_used": doc.get("last_used")
    }

def _log_activity(db, action: str, actor: str, details: dict = None):
    db.db["activity_log"].insert_one({
        "action": action, "actor": actor, "details": details or {},
        "timestamp": datetime.utcnow()
    })

def _get_recent_activity(db, limit: int) -> List[dict]:
    activities = list(db.db["activity_log"].find().sort("timestamp", -1).limit(limit))
    return [{"action": a.get("action"), "actor": a.get("actor"), "details": a.get("details", {}), "timestamp": a.get("timestamp")} for a in activities]
