import requests


def generate_answer(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )

        response.raise_for_status()

        return response.json()["response"]

    except Exception as e:
        return f"⚠ Local LLM error: {str(e)}"
