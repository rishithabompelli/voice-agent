import requests
import json
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"  # Change to phi3:mini if preferred

INTENT_PROMPT = """You are an intent classifier for a voice-controlled AI agent.

Analyze the user's request and return a JSON object with:
- "intents": list of detected intents (can be multiple for compound commands)
- "parameters": relevant extracted parameters for each intent

Supported intents:
1. "create_file"     - user wants to create a file or folder
2. "write_code"      - user wants code written to a file
3. "summarize"       - user wants text summarized
4. "general_chat"    - general conversation or questions

For compound commands (e.g. "write code and save to file"), return multiple intents.

Examples:
Input: "Create a Python file with a retry function"
Output: {"intents": ["write_code", "create_file"], "parameters": {"language": "python", "description": "retry function", "filename": "retry.py"}}

Input: "Summarize this text and save it to summary.txt"
Output: {"intents": ["summarize", "create_file"], "parameters": {"filename": "summary.txt", "save_output": true}}

Input: "What is machine learning?"
Output: {"intents": ["general_chat"], "parameters": {"question": "What is machine learning?"}}

Input: "Create a folder called projects"
Output: {"intents": ["create_file"], "parameters": {"type": "folder", "name": "projects"}}

IMPORTANT: Return ONLY valid JSON. No explanation, no markdown, no extra text.

User request: """


def classify_intent(user_text: str) -> dict:
    """
    Send text to local Ollama LLM and classify intent.
    Returns dict with intents and parameters.
    """
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": INTENT_PROMPT + user_text,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 300
            }
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()

        raw = response.json().get("response", "").strip()

        # Extract JSON safely
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            # Validate structure
            if "intents" not in result:
                result["intents"] = ["general_chat"]
            if "parameters" not in result:
                result["parameters"] = {}
            return result, None
        else:
            return {"intents": ["general_chat"], "parameters": {"raw": raw}}, None

    except requests.exceptions.ConnectionError:
        return None, "Ollama is not running. Please start it with: ollama serve"
    except requests.exceptions.Timeout:
        return None, "Ollama request timed out. Try a smaller model."
    except json.JSONDecodeError as e:
        return {"intents": ["general_chat"], "parameters": {}}, f"JSON parse warning: {e}"
    except Exception as e:
        return None, f"Intent classification error: {str(e)}"


def get_intent_label(intent: str) -> str:
    """Return human-readable label for an intent."""
    labels = {
        "create_file": "📁 Create File/Folder",
        "write_code": "💻 Write Code",
        "summarize": "📝 Summarize Text",
        "general_chat": "💬 General Chat"
    }
    return labels.get(intent, f"❓ {intent}")
