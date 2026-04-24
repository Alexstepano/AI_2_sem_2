#!/usr/bin/env python3
"""Скрипт для тестирования API извне контейнера."""

import argparse
import requests
import json

def test_health(base_url: str = "http://localhost:8000") -> bool:
    """Проверка работоспособности сервиса.
    
    Args:
        base_url: Базовый URL   FastAPI сервиса.

     Returns:
        bool : Статус сервера 
    """
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f" Health: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f" Health check failed: {e}")
        return False

def test_spam_detection(text: str, technique: str = "zero-shot", base_url: str = "http://localhost:8000"):
    """Тестирование эндпоинта классификации.

    Args:
        text : Промт от пользователя, который необходимо проанализировать.
        technique: Техника промптинга, применяемая к запросу
        base_url: Базовый URL   FastAPI сервиса.

    
    
    """
    payload = {"text": text, "technique": technique, "json_output": True}
    try:
        print(f"\n Запрос: {text[:80]}...")
        response = requests.post(f"{base_url}/api/v1/detect", json=payload, timeout=180)
        if response.status_code == 200:
            result = response.json()
            print(f" Вердикт: {' СПАМ' if result['verdict'] == 1 else ' Легитимное'}")
            print(f" {result.get('reasoning', '')}")
            return result
        else:
            print(f" Ошибка {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f" Ошибка: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Текст SMS")
    parser.add_argument("--technique", type=str, default="zero-shot", 
                       choices=["zero-shot", "cot", "few-shot", "cot-few-shot"])
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    args = parser.parse_args()
    
    if not test_health(args.url):
        print(" Убедитесь, что контейнер запущен")
        return
    test_spam_detection(args.text, args.technique, args.url)

if __name__ == "__main__":
    main()
