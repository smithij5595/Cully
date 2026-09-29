"""
Sets up tracing so every agent/model call shows up in Arize AX.

Uses OpenInference (Arize's open tracing convention, built on
OpenTelemetry) plus the arize-otel helper to register a tracer
provider before any agent code runs.

Two instrumentors are wired up, because two different things need
tracing:
  - GoogleGenAIInstrumentor: catches the actual Gemini calls made in
    scoring.py (score_photo talks to the google-genai SDK directly,
    not through LangChain, so this is what captures those spans).
  - LangChainInstrumentor: catches the LangGraph node execution in
    graph.py, so the score -> check flow itself shows up as a trace,
    not just the underlying model call.
"""
import os

from dotenv import load_dotenv

load_dotenv()


_tracing_initialized = False


def init_tracing():
    """
    Call this once, at the top of app.py, before building or running
    the agent graph. Guarded so repeated calls (Streamlit reruns
    app.py on every interaction, in the same process) are no-ops
    after the first.
    """
    global _tracing_initialized
    if _tracing_initialized:
        return

    space_id = os.getenv("ARIZE_SPACE_ID")
    api_key = os.getenv("ARIZE_API_KEY")
    if not space_id or not api_key or space_id == "your_arize_space_id" or api_key == "your_arize_api_key":
      return None

    from arize.otel import register
    from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
    from openinference.instrumentation.langchain import LangChainInstrumentor

    tracer_provider = register(
        space_id=space_id,
        api_key=api_key,
        project_name="photo-culling-agent",
    )
    GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    _tracing_initialized = True
    return tracer_provider