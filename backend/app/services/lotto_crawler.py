import requests
import logging
from app.db import SessionLocal
from app.models.lotto import LottoDraw

logger = logging.getLogger(__name__)


def get_lotto_winning_number(round_num: int = None):
    try:
        if round_num is None:
            lotto_main_page = requests.get("https://dhlottery.co.kr/common.do?method=main").text
            latest_draw = int(lotto_main_page.split('<strong id="lottoDrwNo">')[1].split('</strong>')[0])
            round_num = latest_draw

        response = requests.get(f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={round_num}")
        data = response.json()

        if data["returnValue"] != "success":
            logger.warning(f"{round_num} 회차 당첨 번호를 찾을 수 없습니다.")
            return None

        numbers = [data[f"drwtNo{i}"] for i in range(1, 7)]
        bonus = data["bnusNo"]

        return {
            "draw_no": round_num,
            "draw_date": data["drwNoDate"],
            "numbers": numbers,
            "bonus": bonus
        }
    except Exception as e:
        logger.error(f"당첨 번호 조회 중 오류. 회차 : {round_num}, 내용 : {e}")
        return None


def insert_lotto_data_to_db(start_round=1, end_round=None):
    db = SessionLocal()
    if end_round is None:
        latest_info = get_lotto_winning_number()
        if latest_info is None:
            logger.error("최신 회차를 찾을 수 없습니다.")
            db.close()
            return
        end_round = latest_info["draw_no"]

    for round_num in range(start_round, end_round + 1):
        exists = db.query(LottoDraw).filter(LottoDraw.draw_no == round_num).first()
        if exists:
            continue
        data = get_lotto_winning_number(round_num)
        if not data:
            logger.warning(f"No data for round {round_num}, skipping.")
            continue
        new_draw = LottoDraw(
            draw_no=data["draw_no"],
            drw_no_date=data["draw_date"],
            n1=data["numbers"][0],
            n2=data["numbers"][1],
            n3=data["numbers"][2],
            n4=data["numbers"][3],
            n5=data["numbers"][4],
            n6=data["numbers"][5],
            bonus=data["bonus"]
        )
        db.add(new_draw)
        db.commit()

        print(f"{round_num} 회차 당첨 번호 저장.")

    db.close()
