from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api import lotto
from app.db import engine
from app.models.lotto import Base
from app.services.lotto_crawler import insert_lotto_data_to_db
from app.check_db import check_lotto_data

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    check_lotto_data()
    insert_lotto_data_to_db()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(lotto.router, prefix="/lotto")

@app.get("/")
def root():
    return {"message": "Lotto Recommendation API is running"}
