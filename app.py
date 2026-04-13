import streamlit as st
import os
import tempfile
from pathlib import Path

from stt import load_whisper_model, transcribe_audio
from intent import classify_intent, get_intent_label
from tools import execute_tools
from memory import init_memory, add_to_memory, get_memory_summary, clear_memory

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Voice AI Agent",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-title { font-family: 'JetBrains Mono', monospace; font-size: 2.2rem; font-weight: 700; color: #00D4AA; margin-bottom: 0.2rem; }
    .subtitle { color: #888; font-size: 0.95rem; margin-bottom: 2rem; }
    .result-card { background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 12px; padding: 1.2rem; margin: 0.8rem 0; }
    .result-card h4 { color: #00D4AA; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem; }
    .result-card p { color: #ddd; font-size: 0.95rem; margin: 0; }
    .intent-badge { display: inline-block; background: #0f3460; color: #00D4AA; border: 1px solid #00D4AA; border-radius: 20px; padding: 0.2rem 0.8rem; font-size: 0.8rem; margin: 0.2rem; font-family: 'JetBrains Mono', monospace; }
    .success-box { border-left: 4px solid #00D4AA; padding-left: 1rem; }
    .error-box { border-left: 4px solid #ff4757; padding-left: 1rem; }
    .stButton > button { background: linear-gradient(135deg, #00D4AA, #0f3460); color: white; border: none; border-radius: 8px; padding: 0.6rem 2rem; font-weight: 600; width: 100%; }
    .step-label { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #555; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "whisper_model" not in st.session_state:
    st.session_state.whisper_model = None
if "history" not in st.session_state:
    st.session_state.history = init_memory()
if "transcription" not in st.session_state:
    st.session_state.transcription = None
if "intents" not in st.session_state:
    st.session_state.intents = None
if "parameters" not in st.session_state:
    st.session_state.parameters = None
if "awaiting_confirm" not in st.session_state:
    st.session_state.awaiting_confirm = False
if "results" not in st.session_state:
    st.session_state.results = None

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model_size = st.selectbox("Whisper Model", ["tiny", "base", "small"], index=1)

    if st.button("Load / Reload Model"):
        with st.spinner(f"Loading Whisper-{model_size}..."):
            try:
                st.session_state.whisper_model = load_whisper_model(model_size)
                st.success(f"Whisper-{model_size} loaded!")
            except Exception as e:
                st.error(f"Failed: {e}")

    status = "✅ Loaded" if st.session_state.whisper_model else "❌ Not loaded"
    st.caption(f"Model status: {status}")

    st.divider()
    st.markdown("### 📋 Session History")
    history_text = get_memory_summary(st.session_state.history)
    st.text_area("History", value=history_text, height=200, disabled=True, label_visibility="collapsed")

    if st.button("🗑️ Clear History"):
        st.session_state.history = clear_memory(st.session_state.history)
        st.rerun()

    st.divider()
    st.markdown("### 📁 Output Files")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    files = list(output_dir.iterdir())
    if files:
        for f in files:
            st.caption(f"📄 {f.name}")
    else:
        st.caption("No files yet.")

# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🎙️ Voice AI Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Speak or upload audio → AI detects intent → Executes locally</div>', unsafe_allow_html=True)

# Auto load model
if st.session_state.whisper_model is None:
    with st.spinner("Loading Whisper-base model..."):
        try:
            st.session_state.whisper_model = load_whisper_model("base")
            st.success("Whisper-base loaded!")
        except Exception as e:
            st.error(f"Could not load Whisper: {e}")

# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────
st.markdown("### 🎤 Audio Input")
input_method = st.radio("Choose input method:", ["Upload Audio File", "Record from Microphone"], horizontal=True)

audio_input = None

if input_method == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload .wav or .mp3 file", type=["wav", "mp3", "m4a", "ogg"])
    if uploaded_file:
        st.audio(uploaded_file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.read())
            audio_input = tmp.name

elif input_method == "Record from Microphone":
    try:
        import sounddevice as sd
        from scipy.io.wavfile import write as wav_write
        import numpy as np

        duration = st.slider("Recording duration (seconds)", min_value=3, max_value=15, value=5)
        if st.button("🎙️ Start Recording"):
            with st.spinner(f"Recording for {duration} seconds... Speak now!"):
                sample_rate = 16000
                recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
                sd.wait()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                wav_write(tmp.name, sample_rate, recording)
                st.session_state["mic_audio"] = tmp.name
            st.success("Recording complete!")
            st.audio(st.session_state["mic_audio"])

        if "mic_audio" in st.session_state:
            audio_input = st.session_state["mic_audio"]

    except ImportError:
        st.warning("🎙️ Please install: pip install sounddevice scipy")

st.markdown("#### Or type directly (for testing):")
text_override = st.text_input("Type your command:", placeholder="e.g. Write a Python function for bubble sort")

# ─────────────────────────────────────────────
# PROCESS BUTTON
# ─────────────────────────────────────────────
process_btn = st.button("🚀 Process Command", use_container_width=True)
st.divider()

# ─────────────────────────────────────────────
# STEP 1 & 2: TRANSCRIBE + CLASSIFY
# ─────────────────────────────────────────────
if process_btn and (audio_input or text_override):
    # Reset previous results
    st.session_state.results = None
    st.session_state.awaiting_confirm = False

    st.markdown("### Pipeline Results")
    col_trans, col_intent = st.columns(2)

    # STEP 1: Transcription
    with col_trans:
        st.markdown('<div class="step-label">Step 1 — Transcription</div>', unsafe_allow_html=True)
        with st.spinner("Transcribing..."):
            if text_override:
                transcription = text_override
                source = "Text input"
            else:
                if st.session_state.whisper_model is None:
                    st.error("Whisper model not loaded. Click Load Model in sidebar.")
                    st.stop()
                transcription, error = transcribe_audio(st.session_state.whisper_model, audio_input)
                if error:
                    st.error(f"Transcription failed: {error}")
                    st.stop()
                source = "Whisper-base (local)"

        st.session_state.transcription = transcription
        st.markdown(f"""
        <div class="result-card success-box">
            <h4>Transcribed Text</h4>
            <p>"{transcription}"</p>
            <small style="color:#555">{source}</small>
        </div>
        """, unsafe_allow_html=True)

    # STEP 2: Intent Classification
    with col_intent:
        st.markdown('<div class="step-label">Step 2 — Intent Detection</div>', unsafe_allow_html=True)
        with st.spinner("Classifying intent..."):
            intent_result, error = classify_intent(transcription)
            if error and intent_result is None:
                st.error(f"Intent classification failed: {error}")
                st.stop()

        intents = intent_result.get("intents", ["general_chat"])
        parameters = intent_result.get("parameters", {})

        st.session_state.intents = intents
        st.session_state.parameters = parameters

        badges = "".join([f'<span class="intent-badge">{get_intent_label(i)}</span>' for i in intents])
        st.markdown(f"""
        <div class="result-card success-box">
            <h4>Detected Intents</h4>
            <div>{badges}</div>
            <br>
            <small style="color:#555">Parameters: {str(parameters)[:100]}</small>
        </div>
        """, unsafe_allow_html=True)

    # Check if needs confirmation
    file_intents = [i for i in intents if i in ["create_file", "write_code"]]
    if file_intents:
        st.session_state.awaiting_confirm = True
    else:
        # Execute immediately for non-file intents
        with st.spinner("Executing..."):
            results = execute_tools(intents, parameters, transcription)
        st.session_state.results = results
        st.session_state.history = add_to_memory(st.session_state.history, transcription, intents, results)

# ─────────────────────────────────────────────
# HUMAN-IN-THE-LOOP CONFIRMATION
# ─────────────────────────────────────────────
if st.session_state.awaiting_confirm and st.session_state.transcription:
    file_intents = [i for i in st.session_state.intents if i in ["create_file", "write_code"]]
    st.warning(f"⚠️ This will write files to your filesystem: **{', '.join(file_intents)}**")
    st.markdown("**Do you want to proceed?**")

    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("✅ Yes, Execute", key="confirm_yes"):
            st.session_state.awaiting_confirm = False
            with st.spinner("Executing tools..."):
                results = execute_tools(
                    st.session_state.intents,
                    st.session_state.parameters,
                    st.session_state.transcription
                )
            st.session_state.results = results
            st.session_state.history = add_to_memory(
                st.session_state.history,
                st.session_state.transcription,
                st.session_state.intents,
                results
            )
            st.rerun()

    with col_no:
        if st.button("❌ Cancel", key="confirm_no"):
            st.session_state.awaiting_confirm = False
            st.session_state.transcription = None
            st.info("Operation cancelled.")
            st.rerun()

# ─────────────────────────────────────────────
# STEP 3 & 4: SHOW RESULTS
# ─────────────────────────────────────────────
if st.session_state.results:
    st.markdown("### Pipeline Results")
    st.markdown('<div class="step-label">Step 3 & 4 — Execution Results</div>', unsafe_allow_html=True)

    for result in st.session_state.results:
        intent = result.get("intent", "unknown")
        success = result.get("success", False)
        action = result.get("action", "")
        output = result.get("output", "")
        file_path = result.get("path", "")

        card_class = "success-box" if success else "error-box"
        icon = "✅" if success else "❌"

        st.markdown(f"""
        <div class="result-card {card_class}">
            <h4>{icon} {get_intent_label(intent)}</h4>
            <p><strong>Action:</strong> {action}</p>
            {f'<p><small>📁 Saved to: {file_path}</small></p>' if file_path else ''}
        </div>
        """, unsafe_allow_html=True)

        if output and intent in ["write_code", "summarize", "general_chat"]:
            with st.expander(f"View output — {get_intent_label(intent)}"):
                if intent == "write_code":
                    st.code(output, language="python")
                else:
                    st.markdown(output)

    st.success("✅ Pipeline complete! Check sidebar for output files.")

elif not process_btn and not st.session_state.awaiting_confirm and not st.session_state.results:
    st.markdown("""
    <div class="result-card" style="text-align:center; padding: 3rem;">
        <h4>Ready</h4>
        <p style="font-size:1.1rem">Type a command or upload audio, then click <strong>🚀 Process Command</strong></p>
        <br>
        <p style="color:#555; font-size:0.85rem">Supported: Create files · Write code · Summarize text · General chat · Compound commands</p>
    </div>
    """, unsafe_allow_html=True)
