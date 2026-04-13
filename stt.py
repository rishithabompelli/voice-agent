GROQ_API_KEY = "your_groq_api_key_here"  

def load_whisper_model(model_size="base"):
    print("[STT] Using Groq API")
    return {"type": "groq", "ready": True}

def transcribe_audio(model, audio_input):
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        with open(audio_input, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                response_format="text"
            )
        text = result if isinstance(result, str) else result.text
        return text.strip(), None
    except Exception as e:
        return None, f"Groq error: {str(e)}"