from fastapi import APIRouter, Query, Depends
from typing import Optional
from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.models.lotto import LottoDraw
from app.services.ai_recommender import recommend_by_stat, recommend_by_ml, recommend_by_opt, recommend_by_greedy, recommend_by_dl, recommend_by_hybrid
from app.services.recommender import recommend_numbers

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/recommend")
async def get_recommendation(
    count: int = Query(1, ge=1, le=20),
    exclude: Optional[str] = None,
    include: Optional[str] = None,
    type: str = Query("stat", description="stat|ml|opt"),
    start_round: Optional[int] = Query(None, ge=1),
    end_round: Optional[int] = Query(None, ge=1),
):
    exclude_numbers = [int(x) for x in (exclude or "").split(",") if x.isdigit()]
    include_numbers = [int(x) for x in (include or "").split(",") if x.isdigit()]

    if type == "stat":
        sets = recommend_by_stat(count, exclude_numbers, include_numbers, start_round, end_round)
    elif type == "ml":
        sets = recommend_by_ml(count, exclude_numbers, include_numbers, start_round, end_round)
    elif type == "dl":
        sets = recommend_by_dl(count, exclude_numbers, include_numbers, start_round, end_round)
    elif type == "greedy":
        sets = recommend_by_greedy(count, exclude_numbers, include_numbers, start_round, end_round)
    elif type == "opt":
        sets = recommend_by_opt(count, exclude_numbers, include_numbers, start_round, end_round)
    elif type == "hybrid":
        sets = recommend_by_hybrid(count, exclude, include, start_round, end_round)
    elif type == "normal":
        sets = recommend_numbers(count,exclude_numbers, include_numbers)
    else:
        sets = []

    return {"sets": sets}

@router.get("/history")
def get_lotto_history(db: Session = Depends(get_db)):
    draws = db.query(LottoDraw).order_by(LottoDraw.draw_no.desc()).all()
    return [
        {
            "draw_no": d.draw_no,
            "draw_date": d.drw_no_date,
            "numbers": [d.n1, d.n2, d.n3, d.n4, d.n5, d.n6],
            "bonus": d.bonus,
        }
        for d in draws
    ]

@router.get("/latest-round")
def get_latest_round():
    db = SessionLocal()
    latest = db.query(LottoDraw).order_by(LottoDraw.draw_no.desc()).first()
    db.close()
    if latest:
        return {"latest_round": latest.draw_no}
    return {"latest_round": None}