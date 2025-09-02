from sqlalchemy import Column, Integer, String
from app.db import Base

class LottoDraw(Base):
    __tablename__ = "lotto_draws"

    draw_no = Column(Integer, primary_key=True, index=True)
    drw_no_date = Column(String, nullable=True)
    n1 = Column(Integer, nullable=False)
    n2 = Column(Integer, nullable=False)
    n3 = Column(Integer, nullable=False)
    n4 = Column(Integer, nullable=False)
    n5 = Column(Integer, nullable=False)
    n6 = Column(Integer, nullable=False)
    bonus = Column(Integer, nullable=True)
