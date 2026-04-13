import os
import requests
import json
import re
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def _ollama_generate(prompt: str, max_tokens: int = 1000) -> tuple:
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": max_tokens}
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip(), None
    except Exception as e:
        return None, str(e)


def create_file_or_folder(parameters: dict) -> dict:
    try:
        item_type = parameters.get("type", "file")
        name = parameters.get("name", parameters.get("filename", f"file_{datetime.now().strftime('%H%M%S')}.txt"))
        name = Path(name).name
        target = OUTPUT_DIR / name

        if item_type == "folder":
            target.mkdir(parents=True, exist_ok=True)
            return {"success": True, "action": f"Created folder: output/{name}", "path": str(target), "output": f"Folder 'output/{name}' created successfully."}
        else:
            content = parameters.get("content", f"# Created by Voice Agent\n# {datetime.now()}\n")
            target.write_text(content, encoding="utf-8")
            return {"success": True, "action": f"Created file: output/{name}", "path": str(target), "output": f"File 'output/{name}' created successfully."}
    except Exception as e:
        return {"success": False, "action": "Create file/folder", "output": f"Error: {str(e)}"}


def write_code(parameters: dict, user_text: str) -> dict:
    try:
        language = parameters.get("language", "python")
        description = parameters.get("description", user_text)
        filename = parameters.get("filename", None)

        if not filename:
            ext_map = {"python": ".py", "javascript": ".js", "java": ".java", "html": ".html", "css": ".css", "bash": ".sh", "c": ".c", "cpp": ".cpp"}
            ext = ext_map.get(language.lower(), ".txt")
            safe_name = re.sub(r'[^a-z0-9]', '_', description.lower())[:20]
            filename = f"{safe_name}{ext}"

        filename = Path(filename).name
        target = OUTPUT_DIR / filename

        prompt = f"""Write clean, well-commented {language} code for:
Task: {description}
Write only the code with helpful comments. Make it complete and runnable.
Code:"""

        code, error = _ollama_generate(prompt, max_tokens=800)
        if error:
            return {"success": False, "action": "Write code", "output": f"LLM Error: {error}"}

        code = re.sub(r'^```\w*\n?', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```$', '', code, flags=re.MULTILINE)
        code = code.strip()
        target.write_text(code, encoding="utf-8")

        return {"success": True, "action": f"Generated {language} code saved to output/{filename}", "path": str(target), "output": code}
    except Exception as e:
        return {"success": False, "action": "Write code", "output": f"Error: {str(e)}"}


def summarize_text(parameters: dict, user_text: str) -> dict:
    try:
        text_to_summarize = parameters.get("text", user_text)
        save_to_file = parameters.get("save_output", False)
        filename = parameters.get("filename", None)

        prompt = f"""Provide a clear, concise summary of the following text. Focus on key points.

Text: {text_to_summarize}

Summary:"""

        summary, error = _ollama_generate(prompt, max_tokens=400)
        if error:
            return {"success": False, "action": "Summarize text", "output": f"LLM Error: {error}"}

        result = {"success": True, "action": "Summarized text", "output": summary}

        if save_to_file and filename:
            filename = Path(filename).name
            target = OUTPUT_DIR / filename
            target.write_text(f"# Summary\n\n{summary}\n\n---\nOriginal:\n{text_to_summarize}", encoding="utf-8")
            result["action"] += f" and saved to output/{filename}"
            result["path"] = str(target)

        return result
    except Exception as e:
        return {"success": False, "action": "Summarize", "output": f"Error: {str(e)}"}


def general_chat(parameters: dict, user_text: str) -> dict:
    try:
        prompt = f"""You are a helpful voice-controlled AI assistant.
Answer clearly and concisely.

User: {user_text}
Assistant:"""

        response, error = _ollama_generate(prompt, max_tokens=500)
        if error:
            return {"success": False, "action": "General chat", "output": f"LLM Error: {error}"}

        return {"success": True, "action": "Responded to query", "output": response}
    except Exception as e:
        return {"success": False, "action": "General chat", "output": f"Error: {str(e)}"}


def execute_tools(intents: list, parameters: dict, user_text: str) -> list:
    """Execute all detected intents and return list of results."""
    results = []
    accumulated_output = ""

    for intent in intents:
        if intent == "create_file":
            if accumulated_output:
                parameters["content"] = accumulated_output
            result = create_file_or_folder(parameters)

        elif intent == "write_code":
            result = write_code(parameters, user_text)
            accumulated_output = result.get("output", "")

        elif intent == "summarize":
            result = summarize_text(parameters, user_text)
            accumulated_output = result.get("output", "")

        elif intent == "general_chat":
            result = general_chat(parameters, user_text)

        else:
            result = {"success": False, "action": intent, "output": f"Unknown intent: {intent}"}

        result["intent"] = intent
        results.append(result)

    return results
