from app.db import Base, engine
from app.models import lotto

Base.metadata.create_all(bind=engine)

print("DB 테이블이 생성되었습니다.")