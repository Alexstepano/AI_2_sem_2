"""
Промпты для SMS spam detection — оптимизированы для Qwen2.5:0.5B.

Принципы:
1. Короткие инструкции (малая модель "забывает" длинные контексты и склоняется в сторону всего,как спама)
2. Конкретные примеры (few-shot > абстрактные правила, а также довольно короткие инструкции в cot)
3. Чёткий формат вывода (минимум свободы для галлюцинаций)
4. Английский язык (лучшее качество генерации у Qwen и более качетсвенные ответы за счёт чёткой структуры изначальных примеров)
"""

from typing import Literal
import logging

def get_system_prompt(
    technique: Literal["zero-shot", "cot", "few-shot", "cot-few-shot"],
    json_output: bool = True
) -> str:
    """
    Generate system prompt optimized for Qwen2.5:0.5B.
    
    Key design choices:
    - Keep total prompt < 500 tokens (model's effective context)
    - Put spam examples FIRST (recency bias in small models)
    - Use simple, imperative language
    """
    
     
    ROLE = "You classify SMS as spam(1) or ham(0)."
    
   
    JSON_FMT = '\nOutput JSON only: {"reasoning":"short","verdict":0 or 1}' if json_output else ''
    
    # ========================================================================
    # ZERO-SHOT
    # ========================================================================
    if technique == "zero-shot":
        return f"""{ROLE}

SPAM = 1 if message has: prize/money, "click link", urgency, fake bank/gov, suspicious short URL.
HAM = 0 if personal chat, order update, or normal info.

Message: "{{input}}"

{JSON_FMT}"""

    # ========================================================================
    # COT: минимальное рассуждение в 3 шага
    # ========================================================================
    elif technique == "cot":
        return f"""{ROLE}

Analyze in 3 steps:
1. Does message offer prize/money or ask to "click link"? -> spam signal
2. Is it personal/transactional (order, meeting, family)? -> ham signal  
3. More spam signals? verdict=1. More ham signals? verdict=0.

{JSON_FMT}"""

    # ========================================================================
    # FEW-SHOT: примеры максимально похожи на реальный спам из датасета
    # ========================================================================
    elif technique == "few-shot":
        return f"""{ROLE}


SPAM examples (verdict=1):
- "Won $1000! Click bit.ly/xyz" -> prize + short link
- "URGENT: Verify account: tinyurl.com/abc" -> urgency + fake bank + link
- "XXXClub: click WAP link bellow" -> suspicious brand + typo + link

HAM examples (verdict=0):
- "Hey, dinner at 7pm?" -> personal chat
- "Order #123 shipped: example.com/track" -> legit transactional

Classify this message. Output JSON: {{"reasoning":"SHORT","verdict":0 or 1}}

Message: "{{input}}"
{JSON_FMT}"""

    # ========================================================================
    # COT + FEW-SHOT: структура + примеры
    # ========================================================================
    elif technique == "cot-few-shot":
        return f"""{ROLE}


SPAM examples (verdict=1):
- "Won $1000! Click bit.ly/xyz" -> prize + short link
- "URGENT: Verify account: tinyurl.com/abc" -> urgency + fake bank + link
- "XXXClub: click WAP link bellow" -> suspicious brand + typo + link

HAM examples (verdict=0):
- "Hey, dinner at 7pm?" -> personal chat
- "Order #123 shipped: example.com/track" -> legit transactional

Analyze new message in 3 steps:
1. Does message offer prize/money or ask to "click link"? -> spam signal
2. Is it personal/transactional (order, meeting, family)? -> ham signal  
3. More spam signals? verdict=1. More ham signals? verdict=0.

Message: "{{input}}"
{JSON_FMT}"""
    
    # Fallback
    return get_system_prompt("zero-shot", json_output)


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ 
# =============================================================================

def validate_reasoning(reasoning: str, original_message: str) -> str:
    """Проверка, что reasoning не выдумывает контент.
    
    Args:
        reasoning: Объяснение от  модели
        original_message : Исходное сообщение
            
    Returns:
        Модифицированный выход, отлавливающий сильные галлюцинации
    
    
    
    
    """
    if not reasoning or not original_message:
        return reasoning
    
    msg_lower = original_message.lower()
    reason_lower = reasoning.lower()
    
    # Триггеры галлюцинаций — слова, которых не должно быть в объяснении, 
    # если их нет во входном сообщении(уаведомление от друга о смерти получать не хочется лишний раз)
    hallucination_triggers = [
        "death", "died", "obituary", "inheritance", "funeral",
        "lawsuit", "court", "warrant", "arrest", 
        "bank account frozen", "identity theft"
    ]
    
    for trigger in hallucination_triggers:
        if trigger in reason_lower and trigger not in msg_lower:
            return "reasoning contains unverified claim"
    
    return reasoning


def extract_verdict_from_response(response_text: str, original_message: str = "") -> tuple[int, str]:
    """
    Вытаскиваем вердикт из ответа LLM - на случай галлюцинаций.
    Сначал мы ищем json, потом ищем вердикт, потом ищем ключевые слова(частовстречаемые) и пытаемся хоть так закрыть галлюцинацию модели.

    Args:
        reasoning: Объяснение от  модели
        original_message : Исходное сообщение
            
    Returns:
        Поля JSON, соответствующие возращаемому API


    """
    import json
    import re
    import logging
    
    logger = logging.getLogger(__name__)
    
    if not response_text or not isinstance(response_text, str):
        return 0, "fallback: empty response"
    
     
    response_text = response_text.strip()
    # Пытаемся ответить
    #Попытка 1: найти и распарсить JSON
    try:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response_text[start:end]
            
           
            json_str = (json_str
                .replace('\\"', '"')      # Убираем лишнее экранирование кавычек
                .replace('\\n', ' ')      # Переносы на пробел
                .replace('\\t', ' ')      # Табы на пробел
                .replace('\r', '')        # Убираем \r
            )
            
            data = json.loads(json_str)
            
        
            verdict_key = next((k for k in data.keys() if 'verdict' in k.lower()), 'verdict')
            reasoning_key = next((k for k in data.keys() if 'reasoning' in k.lower()), 'reasoning')
            
            verdict = int(data.get(verdict_key, data.get('verdict', 0)))
            reasoning = str(data.get(reasoning_key, data.get('reasoning', ''))).strip()
            return verdict, reasoning
            
    except (json.JSONDecodeError, ValueError, TypeError, KeyError, AttributeError) as e:
        logger.debug(f"JSON parse failed: {type(e).__name__}: {str(e)[:50]}")
        pass  # Пробуем дальше
    
    #Попытка 2: найти вердикт через regex 
    verdict_match = re.search(r'"?verdict"?\s*[:=]\s*(\d)', response_text, re.IGNORECASE)
    if verdict_match:
        verdict = int(verdict_match.group(1))
        
        reason_match = re.search(r'"?reasoning"?\s*[:=]\s*["\']?([^"\'}\n,]+)', response_text, re.IGNORECASE)
        reasoning = reason_match.group(1).strip() if reason_match else "parsed via regex"
        return verdict, reasoning
    
    #Попытка 3: fallback по ключевым словам во ВХОДНОМ сообщении
    if original_message:
        msg_lower = original_message.lower()
        spam_keywords = [
            'win', 'won', 'prize', 'cash', 'money', '$', '£', '€',
            'click', 'tap', 'link', 'http', 'bit.ly', 'tinyurl', 't.co',
            'urgent', 'act now', 'today only', 'limited time',
            'congratulations', 'selected', 'claim', 'redeem',
            'verify', 'confirm', 'account', 'suspended', 'free', 'gift'
        ]
        is_spam = any(kw in msg_lower for kw in spam_keywords)
        return (1 if is_spam else 0), "fallback: input keywords"
    
    # Дефолт: нейтральный для оценки
    return 1, "fallback: default"