"""
Data Entry routes - manual entry of Title / Description / Skills for keyword extraction
and downstream analysis.
"""
from datetime import datetime
from typing import List, Optional

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.infra.mongodb.repositories.data_entry_repo import (
    get_data_entries_collection,
    get_proposal_training_entries_collection,
)

router = APIRouter()


# ---------- Schemas ----------

class DataEntryIn(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    description: str = Field(..., min_length=1, max_length=20000)
    skills: List[str] = Field(default_factory=list)
    budget: Optional[str] = Field(None, max_length=100)


class DataEntryUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=300)
    description: Optional[str] = Field(None, min_length=1, max_length=20000)
    skills: Optional[List[str]] = None
    budget: Optional[str] = Field(None, max_length=100)


class ProposalTrainingEntryIn(BaseModel):
    job_description: str = Field(..., min_length=1, max_length=20000)
    skills: List[str] = Field(default_factory=list)
    proposal_written: str = Field(..., min_length=1, max_length=30000)
    outcome: str = Field(default="hired")  # "hired" | "discussed"
    invited: bool = Field(default=False)


# ---------- Helpers ----------

def _clean_skills(skills: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in skills or []:
        v = (s or "").strip()
        if not v:
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _serialize(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "title": doc.get("title", ""),
        "description": doc.get("description", ""),
        "skills": doc.get("skills", []),
        "budget": doc.get("budget"),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


def _serialize_proposal_training_entry(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "job_description": doc.get("job_description", ""),
        "skills": doc.get("skills", []),
        "proposal_written": doc.get("proposal_written", ""),
        "outcome": doc.get("outcome", "hired"),
        "invited": doc.get("invited", False),
        "created_at": doc.get("created_at"),
    }


# ---------- Routes ----------

@router.post("/data-entries")
def create_entry(payload: DataEntryIn):
    skills = _clean_skills(payload.skills)
    if not skills:
        raise HTTPException(status_code=400, detail="At least one skill is required")

    now = datetime.utcnow().isoformat()
    doc = {
        "title": payload.title.strip(),
        "description": payload.description.strip(),
        "skills": skills,
        "budget": (payload.budget or "").strip() or None,
        "created_at": now,
        "updated_at": now,
    }
    result = get_data_entries_collection().insert_one(doc)
    if not result.inserted_id:
        raise HTTPException(status_code=500, detail="Failed to save entry")
    doc["_id"] = result.inserted_id
    return {"message": "Saved successfully", "entry": _serialize(doc)}


@router.get("/data-entries")
def list_entries(
    limit: int = Query(100, ge=1, le=500),
    skip: int = Query(0, ge=0),
    search: Optional[str] = Query(None, max_length=200),
):
    coll = get_data_entries_collection()
    query: dict = {}
    if search:
        # Case-insensitive substring match across key text fields
        # Escape regex specials defensively
        import re
        pattern = re.escape(search.strip())
        if pattern:
            query = {
                "$or": [
                    {"title": {"$regex": pattern, "$options": "i"}},
                    {"description": {"$regex": pattern, "$options": "i"}},
                    {"skills": {"$regex": pattern, "$options": "i"}},
                ]
            }

    total = coll.count_documents(query)
    cursor = coll.find(query).sort("_id", -1).skip(skip).limit(limit)
    items = [_serialize(d) for d in cursor]
    return {"items": items, "count": len(items), "total": total}


@router.get("/data-entries/{entry_id}")
def get_entry(entry_id: str):
    try:
        oid = ObjectId(entry_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="Invalid id")
    doc = get_data_entries_collection().find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")
    return _serialize(doc)


@router.put("/data-entries/{entry_id}")
def update_entry(entry_id: str, payload: DataEntryUpdate):
    try:
        oid = ObjectId(entry_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="Invalid id")

    update: dict = {}
    if payload.title is not None:
        update["title"] = payload.title.strip()
    if payload.description is not None:
        update["description"] = payload.description.strip()
    if payload.skills is not None:
        cleaned = _clean_skills(payload.skills)
        if not cleaned:
            raise HTTPException(status_code=400, detail="At least one skill is required")
        update["skills"] = cleaned
    if payload.budget is not None:
        update["budget"] = payload.budget.strip() or None

    if not update:
        raise HTTPException(status_code=400, detail="No fields to update")

    update["updated_at"] = datetime.utcnow().isoformat()

    coll = get_data_entries_collection()
    result = coll.find_one_and_update(
        {"_id": oid}, {"$set": update}, return_document=True
    )
    if not result:
        raise HTTPException(status_code=404, detail="Not found")
    return {"message": "Updated", "entry": _serialize(result)}


@router.delete("/data-entries/{entry_id}")
def delete_entry(entry_id: str):
    try:
        oid = ObjectId(entry_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="Invalid id")
    result = get_data_entries_collection().delete_one({"_id": oid})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Not found")
    return {"message": "Deleted"}


@router.post("/proposal-training-entries")
def create_proposal_training_entry(payload: ProposalTrainingEntryIn):
    skills = _clean_skills(payload.skills)
    if not skills:
        raise HTTPException(status_code=400, detail="At least one skill is required")

    now = datetime.utcnow().isoformat()
    outcome = payload.outcome if payload.outcome in ("hired", "discussed") else "hired"
    doc = {
        "job_description": payload.job_description.strip(),
        "skills": skills,
        "proposal_written": payload.proposal_written.strip(),
        "outcome": outcome,
        "invited": bool(payload.invited),
        "created_at": now,
    }
    result = get_proposal_training_entries_collection().insert_one(doc)
    if not result.inserted_id:
        raise HTTPException(status_code=500, detail="Failed to save proposal training entry")
    doc["_id"] = result.inserted_id
    return {
        "message": "Saved successfully",
        "entry": _serialize_proposal_training_entry(doc),
    }


@router.get("/proposal-training-entries")
def list_proposal_training_entries(
    limit: int = Query(100, ge=1, le=500),
    skip: int = Query(0, ge=0),
    search: Optional[str] = Query(None, max_length=200),
):
    coll = get_proposal_training_entries_collection()
    query: dict = {}
    if search:
        import re

        pattern = re.escape(search.strip())
        if pattern:
            query = {
                "$or": [
                    {"job_description": {"$regex": pattern, "$options": "i"}},
                    {"skills": {"$regex": pattern, "$options": "i"}},
                    {"proposal_written": {"$regex": pattern, "$options": "i"}},
                ]
            }

    total = coll.count_documents(query)
    cursor = coll.find(query).sort("_id", -1).skip(skip).limit(limit)
    items = [_serialize_proposal_training_entry(d) for d in cursor]
    return {"items": items, "count": len(items), "total": total}


@router.put("/proposal-training-entries/{entry_id}")
def update_proposal_training_entry(entry_id: str, payload: ProposalTrainingEntryIn):
    try:
        oid = ObjectId(entry_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="Invalid id")

    skills = _clean_skills(payload.skills)
    if not skills:
        raise HTTPException(status_code=400, detail="At least one skill is required")

    outcome = payload.outcome if payload.outcome in ("hired", "discussed") else "hired"
    result = get_proposal_training_entries_collection().find_one_and_update(
        {"_id": oid},
        {"$set": {
            "job_description": payload.job_description.strip(),
            "skills": skills,
            "proposal_written": payload.proposal_written.strip(),
            "outcome": outcome,
            "invited": bool(payload.invited),
        }},
        return_document=True,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Not found")
    return {"message": "Updated", "entry": _serialize_proposal_training_entry(result)}


@router.delete("/proposal-training-entries/{entry_id}")
def delete_proposal_training_entry(entry_id: str):
    try:
        oid = ObjectId(entry_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="Invalid id")
    result = get_proposal_training_entries_collection().delete_one({"_id": oid})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Not found")
    return {"message": "Deleted"}
