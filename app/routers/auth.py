from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
import bcrypt

from app.models import Manager
from app.database import SessionLocal

router = APIRouter(prefix="/auth", tags=["auth"])

# ✅ Updated SignupRequest model (removed confirm_password)
class SignupRequest(BaseModel):
    name: str
    hall: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/signup")
def signup(data: SignupRequest, db: Session = Depends(get_db)):
    email = data.email.lower()

    if len(data.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    existing = db.query(Manager).filter(Manager.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered.")

    hashed_password = bcrypt.hashpw(data.password.encode('utf-8'), bcrypt.gensalt())
    manager = Manager(
        name=data.name,
        hall=data.hall,
        email=email,
        hashed_password=hashed_password.decode('utf-8')
    )
    db.add(manager)
    db.commit()
    db.refresh(manager)

    return {"message": "Signup successful"}

@router.post("/login")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    email = data.email.lower()

    manager = db.query(Manager).filter(Manager.email == email).first()

    if not manager or not bcrypt.checkpw(data.password.encode('utf-8'), manager.hashed_password.encode('utf-8')):
        raise HTTPException(status_code=400, detail="Invalid email or password")

    return {
        "message": "Login successful",
        "name": manager.name,
        "hall": manager.hall,
        "email": manager.email
    }
