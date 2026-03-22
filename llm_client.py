import os
import openai

from logger import logger

class OpenRouterLLM:
    def __init__(self, model: str = 'openai/gpt-oss-120b:free'):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        self.model = model 

        self.base_url = "https://openrouter.ai/api/v1"

    def generate(self, prompt: str) -> str:
        system_prompt = """Ты — помощник по работе с документами.\n
            Используй ТОЛЬКО информацию из приведённых ниже фрагментов. Если ответа в фрагментах нет — так и скажи.
            Правила:\n
            - Не используй внешние знания.\n
            - Не делай предположений.\n
            - Не дополняй ответ от себя.\n
            - Отвечай строго на русском языке."""
        try:
            client = openai.OpenAI(api_key=self.api_key,
                                base_url=self.base_url)
            response = client.chat.completions.create(
                messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                model=self.model,
                temperature=0,
                timeout=180
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Ошибка при обработке данных: {e}")
            raise