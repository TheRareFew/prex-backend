from fastapi import APIRouter, HTTPException
from app.models.schemas import Query, Response

router = APIRouter()

@router.post("/predict", response_model=Response)
async def predict(query: Query):
    try:
        # Add your AI logic here
        return Response(response="Test response")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 