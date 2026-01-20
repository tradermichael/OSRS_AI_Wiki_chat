from __future__ import annotations

from dataclasses import dataclass

from ..core.config import settings


@dataclass(frozen=True)
class GeminiResult:
    text: str


class GeminiVertexClient:
    def __init__(self, *, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.gemini_model
        self._project = settings.google_cloud_project
        self._location = settings.vertex_location

    def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> GeminiResult:
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

        t = 0.2 if temperature is None else float(temperature)
        mot = 2048 if max_output_tokens is None else int(max_output_tokens)

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": t,
                "max_output_tokens": mot,
            },
        )

        text = getattr(response, "text", None)
        if not text:
            # Some SDK versions nest candidates; keep a conservative fallback
            text = str(response)

        return GeminiResult(text=text)
