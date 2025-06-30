from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import complaint, auth
from app.ai_model import load_models

# 🆕 DB setup
from app.database import Base, engine
from app import models

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(complaint.router)
app.include_router(auth.router)

# On startup, load models and initialize DB
@app.on_event("startup")
async def startup_event():
    load_models()
    Base.metadata.create_all(bind=engine)
    print("✅ AI models loaded and database tables created.")
