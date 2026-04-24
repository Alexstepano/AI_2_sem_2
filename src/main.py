"""FastAPI сервис для обёртки Ollama сервера."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
import logging
import requests
import json

from .ollama_client import OllamaClient
from .prompts import get_system_prompt, extract_verdict_from_response, validate_reasoning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SMS Spam Detection API", version="1.0.0")
ollama_client = OllamaClient(base_url="http://localhost:11434")


class SpamDetectionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    technique: Literal["zero-shot", "cot", "few-shot", "cot-few-shot"] = "zero-shot"
    json_output: bool = True


class SpamDetectionResponse(BaseModel):
    verdict: Literal[0, 1]
    reasoning: Optional[str] = None
    model_used: str


@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса."""
    ollama_ok = ollama_client.check_health()
    return {"status": "healthy" if ollama_ok else "degraded", "ollama_available": ollama_ok}


@app.post("/api/v1/detect", response_model=SpamDetectionResponse)
async def detect_spam(request: SpamDetectionRequest):
    """Основной эндпоинт для классификации SMS."""
    try:
        # 1. Получить промпт
        system_prompt = get_system_prompt(
            technique=request.technique,
            json_output=request.json_output
        )
        
        # 2. Сформировать запрос
        
        
        system_prompt = get_system_prompt(request.technique, request.json_output)
        user_prompt = f"Message: \"{request.text}\""  
        # 3. Вызвать Ollama
        response = ollama_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            stream=False,
            temperature=0.0,  # Минимум случайности
            options={"num_predict": 128} 
        )

         
        raw_response = response.get("response", "")
        logger.info(f"=== RAW MODEL OUTPUT ===") 
        logger.info(f"Input: {request.text[:100]}")
        logger.info(f"Technique: {request.technique}")
        logger.info(f"Raw response: {raw_response[:500]}")  
        logger.info(f"=== END RAW OUTPUT ===")
        
        # 4. Надёжно распарсить ответ
        verdict, reasoning = extract_verdict_from_response(
        response.get("response", ""), 
        original_message=request.text  
        )
        
        
        # 5. Валидировать reasoning (опционально, для аудита)
        reasoning = validate_reasoning(reasoning, request.text)
        
        return SpamDetectionResponse(
            verdict=verdict,
            reasoning=reasoning,
            model_used=ollama_client.model_name
        )
        
    except Exception as e:
        logger.error(f"Detection error: {type(e).__name__}: {str(e)[:100]}")
        logger.debug(f"Full error context: text={request.text[:100]}, technique={request.technique}")
        # Fallback-вердикт: при ошибке лучше пометить как спам(имхо, зависит от задачи.)
        return SpamDetectionResponse(
            verdict=1,
            reasoning="ERRR: fallback to spam verdict",
            model_used=ollama_client.model_name
        )
