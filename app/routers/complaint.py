from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List

from app.models import Complaint
from app.database import SessionLocal
from app.ai_model import predict_complaint_metadata

router = APIRouter(prefix="/complaints", tags=["complaints"])

class ComplaintRequest(BaseModel):
    name: str
    roll_number: str
    room_number: str
    hall: str
    description: str

class StatusUpdateRequest(BaseModel):
    status: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/")
async def submit_complaint(data: ComplaintRequest, db: Session = Depends(get_db)):
    normalized_hall = data.hall.strip().lower()
    category, urgency, location, summary = predict_complaint_metadata(
        data.room_number, data.description
    )

    complaint = Complaint(
        name=data.name,
        roll_number=data.roll_number,
        room_number=data.room_number,
        hall=normalized_hall,
        description=data.description,
        category=category,
        urgency=urgency,
        location=location,
        summary=summary,
        status="Pending"
    )
    db.add(complaint)
    db.commit()
    db.refresh(complaint)
    return {"message": "Complaint submitted successfully"}

@router.get("/")
def get_complaints(authorization: str = Header(None), db: Session = Depends(get_db)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    hall = authorization.replace("Bearer ", "").strip().lower()

    complaints = (
        db.query(Complaint)
        .filter(func.lower(func.trim(Complaint.hall)) == hall)
        .order_by(Complaint.id.desc())
        .all()
    )

    return [
        {
            "id": c.id,
            "name": c.name,
            "roll_number": c.roll_number,
            "room_number": c.room_number,
            "description": c.description,
            "category": c.category,
            "urgency": c.urgency,
            "location": c.location,
            "summary": c.summary,
            "status": c.status,
        }
        for c in complaints
    ]

@router.patch("/status/{complaint_id}")
def update_complaint_status(
    complaint_id: int,
    data: StatusUpdateRequest,
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    hall = authorization.replace("Bearer ", "").strip().lower()

    complaint = db.query(Complaint).filter(
        Complaint.id == complaint_id,
        func.lower(func.trim(Complaint.hall)) == hall
    ).first()

    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")

    complaint.status = data.status
    db.commit()
    return {"message": "Status updated"}

@router.delete("/{complaint_id}")
def delete_complaint(
    complaint_id: int,
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    hall = authorization.replace("Bearer ", "").strip().lower()

    complaint = db.query(Complaint).filter(
        Complaint.id == complaint_id,
        func.lower(func.trim(Complaint.hall)) == hall
    ).first()

    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")

    db.delete(complaint)
    db.commit()
    return {"message": "Complaint deleted successfully"}
