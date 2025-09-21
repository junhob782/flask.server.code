from fastapi import APIRouter, HTTPException
from spot.services.parking_service import ParkingService

router = APIRouter(prefix="/api/parking", tags=["parking"])
service = ParkingService()

@router.get("/vacancies")
async def get_vacancies():
    """
    현재 프레임 기준 모든 슬롯의 점유 상태를 JSON으로 반환
    """
    try:
        status = service.get_current_slot_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return status