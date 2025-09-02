from app.db import SessionLocal
from app.models.lotto import LottoDraw

def check_lotto_data():
    db = SessionLocal()
    count = db.query(LottoDraw).count()
    print(f"총 로또 데이터 개수: {count}")
    if count > 0:
        latest = db.query(LottoDraw).order_by(LottoDraw.draw_no.desc()).first()
        print(f"최신 회차: {latest.draw_no}, 번호: {[latest.n1, latest.n2, latest.n3, latest.n4, latest.n5, latest.n6]}")
    else:
        print("로또 데이터가 없습니다.")
    db.close()

if __name__ == "__main__":
    check_lotto_data()
