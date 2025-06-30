from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import complaint, auth
from app.ai_model import load_models

# 🆕 Add these
from app.database import Base, engine
from app import models

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(complaint.router)
app.include_router(auth.router)

# Load models & initialize DB on startup
@app.on_event("startup")
async def startup_event():
    load_models()
    Base.metadata.create_all(bind=engine)  # ✅ Recreate DB tables
    print("✅ AI models loaded and tables created.")
