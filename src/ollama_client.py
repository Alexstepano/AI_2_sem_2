"""Клиент для взаимодействия с Ollama сервером."""

import requests
import json
from typing import Optional, Dict, Any


class OllamaClient:
    """Клиент для общения с локальным сервером Ollama."""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen2.5:0.5b",
        timeout: int = 120
    ):
        """Инициализация клиента Ollama."""
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.generate_endpoint = f"{self.base_url}/api/generate"
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        stream: bool = False,
        temperature: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Отправка запроса на генерацию текста через Ollama."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": 128
            }
        }
        if system_prompt:
            payload["system"] = system_prompt
            
        response = requests.post(
            self.generate_endpoint,
            json=payload,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    
    def check_health(self) -> bool:
        """Проверка доступности сервера Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except requests.RequestException:
            return False
