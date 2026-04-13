from datetime import datetime


def init_memory() -> list:
    """Initialize empty session memory."""
    return []


def add_to_memory(history: list, transcription: str, intents: list, results: list) -> list:
    """Add a completed interaction to session memory."""
    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "transcription": transcription,
        "intents": intents,
        "results": results,
        "success": all(r.get("success", False) for r in results)
    }
    history.append(entry)
    return history


def get_memory_summary(history: list) -> str:
    """Return a readable summary of session history."""
    if not history:
        return "No actions taken yet in this session."

    lines = []
    for i, entry in enumerate(history, 1):
        status = "✅" if entry["success"] else "❌"
        intents_str = ", ".join(entry["intents"])
        lines.append(f"{status} [{entry['timestamp']}] #{i}: {entry['transcription'][:60]}... → {intents_str}")

    return "\n".join(lines)


def get_context_for_llm(history: list, max_entries: int = 3) -> str:
    """Return recent history context to pass to LLM for continuity."""
    if not history:
        return ""

    recent = history[-max_entries:]
    lines = ["Recent actions for context:"]
    for entry in recent:
        lines.append(f"- User said: '{entry['transcription']}'")
        lines.append(f"  Actions taken: {', '.join(entry['intents'])}")

    return "\n".join(lines)


def clear_memory(history: list) -> list:
    """Clear all session memory."""
    history.clear()
    return history
