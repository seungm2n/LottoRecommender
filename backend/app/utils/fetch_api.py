import httpx

async def fetch_lotto_result(drw_no: int) -> dict:
    url = f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={drw_no}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()