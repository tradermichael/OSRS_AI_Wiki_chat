from __future__ import annotations

from dataclasses import dataclass

from ..core.config import settings


@dataclass(frozen=True)
class GeminiResult:
    text: str


class GeminiVertexClient:
    def __init__(self) -> None:
        self._model_name = settings.gemini_model
        self._project = settings.google_cloud_project
        self._location = settings.vertex_location

    def generate(self, prompt: str) -> GeminiResult:
        if not self._project:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT is not set. Configure Vertex AI credentials and set GOOGLE_CLOUD_PROJECT."
            )

        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing Vertex AI SDK. Install google-cloud-aiplatform and ensure imports work."
            ) from exc

        vertexai.init(project=self._project, location=self._location)
        model = GenerativeModel(self._model_name)

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 1024,
            },
        )

        text = getattr(response, "text", None)
        if not text:
            # Some SDK versions nest candidates; keep a conservative fallback
            text = str(response)

        return GeminiResult(text=text)
