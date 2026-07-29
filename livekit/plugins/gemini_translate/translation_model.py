"""
Standalone translation support for Google's ``gemini-3.5-live-translate-preview`` model.

This module subclasses ``livekit-plugins-google``'s realtime ``RealtimeModel`` /
``RealtimeSession`` to add speech-to-speech translation support without modifying
the original plugin.

Unlike the OpenAI translate endpoint (which speaks a bespoke event protocol),
Gemini Live Translate reuses the *same* Gemini Live ``server_content`` wire format
as the regular realtime model — translated audio arrives as
``model_turn.parts[].inline_data``, the source transcript as
``input_transcription`` and the translated transcript as ``output_transcription``.
The base ``RealtimeSession`` already decodes all of those, so the only behaviour we
override is the connect config: we point the session at the translate model and
inject a ``translation_config`` (target language + echo behaviour), while dropping
the prompt / voice / tool fields the translate endpoint does not accept.

Usage:
    from livekit.plugins.gemini_translate import GeminiTranslationModel

    model = GeminiTranslationModel(
        api_key="your-api-key",
        target_language="pl",
        echo_target_language=True,
    )
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from google.genai import types
from livekit import rtc
from livekit.agents import llm
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from livekit.plugins.google.realtime.realtime_api import RealtimeModel, RealtimeSession

logger = logging.getLogger(__name__)

# Dedicated Gemini Live speech-to-speech translation model.
# https://ai.google.dev/gemini-api/docs/live-api/live-translate
TRANSLATION_MODEL = "gemini-3.5-live-translate-preview"

# When the incoming speech is already in the target language, echo it verbatim
# (True) instead of staying silent (False). Mirrors the docs default.
DEFAULT_ECHO_TARGET_LANGUAGE = True

# Seconds of silence in the transcript stream after which the current utterance is
# finalized as its own conversation item. The translate model streams continuously
# and does not send a `turn_complete` per utterance, so — like the OpenAI/Azure
# translate plugins — we cut segments on silence instead of on turn boundaries.
DEFAULT_SEGMENT_FLUSH_DELAY = 0.6

# Sentence-ending characters used to split continuous translated speech into one
# conversation item per sentence (Latin, Devanagari danda, CJK, Arabic, ellipsis).
# Silence-flushing alone can't split speech that never pauses; the translated
# transcript's own punctuation is the reliable per-sentence boundary.
_SENTENCE_END_CHARS = "।॥.?!…؟。！？"


def _coerce_start_sensitivity(
    value: "types.StartSensitivity | str",
) -> types.StartSensitivity:
    """Accept a ``StartSensitivity`` enum, ``"HIGH"``/``"LOW"``, or the full enum name."""
    if isinstance(value, types.StartSensitivity):
        return value
    key = str(value).strip().upper()
    if not key.startswith("START_SENSITIVITY_"):
        key = f"START_SENSITIVITY_{key}"
    return types.StartSensitivity[key]


def _coerce_end_sensitivity(
    value: "types.EndSensitivity | str",
) -> types.EndSensitivity:
    """Accept an ``EndSensitivity`` enum, ``"HIGH"``/``"LOW"``, or the full enum name."""
    if isinstance(value, types.EndSensitivity):
        return value
    key = str(value).strip().upper()
    if not key.startswith("END_SENSITIVITY_"):
        key = f"END_SENSITIVITY_{key}"
    return types.EndSensitivity[key]


class GeminiTranslationModel(RealtimeModel):
    """
    RealtimeModel subclass for ``gemini-3.5-live-translate-preview``.

    Connects to the Gemini Live API with a ``translation_config`` so the model
    performs cascaded ASR -> translate -> TTS, emitting translated audio plus the
    source and translated transcripts. Behaves like the OpenAI / Azure translate
    providers and plugs into the same ``RealtimeTranslator`` pipeline.

    Args:
        target_language (str): BCP-47 language code for the spoken output, e.g.
            "hi", "es", "pl", "fr".
        echo_target_language (bool): If True, speech already in the target language
            is echoed back verbatim; if False the model stays silent for it.
        model (str): Translate model name. Defaults to ``gemini-3.5-live-translate-preview``.
        api_key (str): Google Gemini API key (or set ``GOOGLE_API_KEY``). Not
            required when ``vertexai=True``.
        input_audio_transcription: Config for the source-language transcript.
            Defaults to an enabled ``AudioTranscriptionConfig()``.
        output_audio_transcription: Config for the translated transcript.
            Defaults to an enabled ``AudioTranscriptionConfig()``.
        realtime_input_config: Full ``types.RealtimeInputConfig`` passthrough. When
            given it takes precedence over the ``vad_*`` convenience args below.
        vad_silence_duration_ms (int): End-of-speech silence, in ms, before the model
            commits the turn and starts producing translated audio. **The single
            biggest latency lever** — lower is snappier (e.g. 100-300) at the risk of
            clipping speakers who pause mid-thought. ``None`` keeps the server default.
        vad_prefix_padding_ms (int): Audio kept before detected speech start, in ms.
        vad_start_sensitivity / vad_end_sensitivity: How eagerly the server VAD
            fires at speech start / end. Accepts the ``types.StartSensitivity`` /
            ``types.EndSensitivity`` enums or the strings ``"HIGH"`` / ``"LOW"``.
            ``"HIGH"`` end-sensitivity detects end-of-speech faster (lower latency).
        context_window_compression: Optional ``ContextWindowCompressionConfig``
            passthrough.
        vertexai / project / location / credentials: VertexAI wiring, forwarded to
            the base Google realtime model.
        conn_options / http_options / api_version: forwarded to the base model.

    Example:
        from livekit.plugins.gemini_translate import GeminiTranslationModel

        model = GeminiTranslationModel(
            api_key="...",
            target_language="hi",
            echo_target_language=True,
        )
    """

    def __init__(
        self,
        *,
        target_language: str,
        echo_target_language: bool = DEFAULT_ECHO_TARGET_LANGUAGE,
        segment_flush_delay: float | None = None,
        split_on_sentence: bool = True,
        realtime_input_config: NotGivenOr[types.RealtimeInputConfig] = NOT_GIVEN,
        vad_silence_duration_ms: int | None = None,
        vad_prefix_padding_ms: int | None = None,
        vad_start_sensitivity: "types.StartSensitivity | str | None" = None,
        vad_end_sensitivity: "types.EndSensitivity | str | None" = None,
        context_window_compression: NotGivenOr[
            types.ContextWindowCompressionConfig
        ] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[types.AudioTranscriptionConfig | None] = NOT_GIVEN,
        output_audio_transcription: NotGivenOr[types.AudioTranscriptionConfig | None] = NOT_GIVEN,
        vertexai: NotGivenOr[bool] = NOT_GIVEN,
        project: NotGivenOr[str] = NOT_GIVEN,
        location: NotGivenOr[str] = NOT_GIVEN,
        credentials: Any = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        http_options: NotGivenOr[types.HttpOptions] = NOT_GIVEN,
        api_version: NotGivenOr[str] = NOT_GIVEN,
        **kwargs: Any,
    ) -> None:
        # The translate config surface (`types.TranslationConfig`) only exists on
        # recent google-genai releases — fail fast with an actionable message.
        if not hasattr(types, "TranslationConfig"):
            raise RuntimeError(
                "The installed google-genai does not expose `types.TranslationConfig`. "
                "Gemini Live Translate requires google-genai>=1.55; please upgrade."
            )

        # Force the translate model. Translation only works on this dedicated model,
        # so a wrong/blank config `model` (e.g. a chat model, or the id without its
        # required "-preview" suffix) must never be used — that produces a 1008
        # "model not found / not supported for bidiGenerateContent" at connect time.
        translate_model = TRANSLATION_MODEL
        if is_given(model) and model and model != TRANSLATION_MODEL:
            logger.warning(
                "GeminiTranslationModel ignoring model=%r; translation requires %r",
                model,
                TRANSLATION_MODEL,
            )

        # Translate model only speaks audio; enable both transcripts by default so
        # the source + translated text is surfaced as conversation items.
        if not is_given(input_audio_transcription):
            input_audio_transcription = types.AudioTranscriptionConfig()
        if not is_given(output_audio_transcription):
            output_audio_transcription = types.AudioTranscriptionConfig()

        # Resolve the server-side VAD / turn-detection config. An explicit
        # `realtime_input_config` always wins; otherwise assemble one from the
        # `vad_*` convenience args. Lowering `silence_duration_ms` and using HIGH
        # end-of-speech sensitivity is the main way to cut translation latency —
        # the model commits the turn sooner and starts speaking the translation.
        # If nothing is specified we send no config and keep the server defaults,
        # so the default behaviour is unchanged and safe.
        resolved_input_config = realtime_input_config
        if not is_given(resolved_input_config):
            aad_kwargs: dict[str, Any] = {}
            if vad_silence_duration_ms is not None:
                aad_kwargs["silence_duration_ms"] = int(vad_silence_duration_ms)
            if vad_prefix_padding_ms is not None:
                aad_kwargs["prefix_padding_ms"] = int(vad_prefix_padding_ms)
            if vad_start_sensitivity is not None:
                aad_kwargs["start_of_speech_sensitivity"] = _coerce_start_sensitivity(
                    vad_start_sensitivity
                )
            if vad_end_sensitivity is not None:
                aad_kwargs["end_of_speech_sensitivity"] = _coerce_end_sensitivity(
                    vad_end_sensitivity
                )
            if aad_kwargs:
                resolved_input_config = types.RealtimeInputConfig(
                    automatic_activity_detection=types.AutomaticActivityDetection(
                        **aad_kwargs
                    )
                )

        super().__init__(
            model=translate_model,
            api_key=api_key,
            # translate endpoint is audio-only
            modalities=[types.Modality.AUDIO],
            input_audio_transcription=input_audio_transcription,
            output_audio_transcription=output_audio_transcription,
            realtime_input_config=resolved_input_config,
            context_window_compression=context_window_compression,
            vertexai=vertexai,
            project=project,
            location=location,
            credentials=credentials,
            conn_options=conn_options,
            http_options=http_options,
            api_version=api_version,
        )

        self._target_language = target_language
        self._echo_target_language = echo_target_language
        self._segment_flush_delay = (
            float(segment_flush_delay)
            if segment_flush_delay is not None
            else DEFAULT_SEGMENT_FLUSH_DELAY
        )
        self._split_on_sentence = split_on_sentence

    @property
    def target_language(self) -> str:
        return self._target_language

    @property
    def echo_target_language(self) -> bool:
        return self._echo_target_language

    @property
    def segment_flush_delay(self) -> float:
        return self._segment_flush_delay

    @property
    def split_on_sentence(self) -> bool:
        return self._split_on_sentence

    def session(self) -> GeminiTranslationSession:  # type: ignore[override]
        sess = GeminiTranslationSession(self)
        self._sessions.add(sess)
        return sess


class GeminiTranslationSession(RealtimeSession):
    """
    RealtimeSession subclass that drives the ``gemini-3.5-live-translate-preview`` model.

    Inherits the base Google realtime session end-to-end — audio ingestion,
    resampling, reconnect/retry, and ``server_content`` decoding (translated audio,
    source + translated transcripts). The differences from the base session are:

    - the connect config carries a ``translation_config`` and drops the fields the
      translate endpoint rejects (prompt, voice/speech config, tools);
    - the model has no prompt or tools, so instruction / chat-context / tool updates
      are no-ops and ``generate_reply`` is disabled (translation is passive).
    """

    def __init__(self, realtime_model: GeminiTranslationModel) -> None:
        # Set before super().__init__: the base ctor schedules `_main_task`, which
        # calls our `_build_connect_config`.
        self._translation_model = realtime_model
        self._segment_flush_delay = realtime_model.segment_flush_delay
        self._split_on_sentence = realtime_model.split_on_sentence
        self._segment_flush_handle: asyncio.TimerHandle | None = None
        super().__init__(realtime_model)

    # ------------------------------------------------------------------
    # Override: connect config — translate model + translation_config
    # ------------------------------------------------------------------

    def _build_connect_config(self) -> types.LiveConnectConfig:
        opts = self._opts

        translation_config = types.TranslationConfig(
            target_language_code=self._translation_model.target_language,
            echo_target_language=self._translation_model.echo_target_language,
        )

        conf = types.LiveConnectConfig(
            response_modalities=opts.response_modalities,
            input_audio_transcription=opts.input_audio_transcription,
            output_audio_transcription=opts.output_audio_transcription,
            translation_config=translation_config,
        )

        # Optional passthroughs the translate endpoint supports.
        if self._session_resumption_handle:
            conf.session_resumption = types.SessionResumptionConfig(
                handle=self._session_resumption_handle
            )
        # Server-side VAD / turn-detection tuning — the main latency lever.
        if is_given(opts.realtime_input_config):
            conf.realtime_input_config = opts.realtime_input_config
        if is_given(opts.context_window_compression):
            conf.context_window_compression = opts.context_window_compression

        return conf

    # ------------------------------------------------------------------
    # Override: per-utterance segmentation
    #
    # The base session only finalizes a conversation item on the server's
    # `turn_complete`. The translate model streams continuously and does not send
    # a `turn_complete` per utterance, so without this every utterance would be
    # appended into a single item. We finalize the current item on a strong
    # boundary (`generation_complete`) or after a short silence in the transcript
    # stream — the same approach the OpenAI/Azure translate plugins use.
    # ------------------------------------------------------------------

    def _handle_server_content(self, server_content: types.LiveServerContent) -> None:
        super()._handle_server_content(server_content)

        gen = self._current_generation
        if gen is None or gen._done:
            # base already finalized this item (e.g. turn_complete) — nothing to cut
            return

        # Strong boundary: the model finished a response generation → cut now.
        if server_content.generation_complete:
            self._flush_segment()
            return

        # Sentence boundary in the translated transcript → cut per sentence. This
        # splits continuous speech (which never goes silent mid-turn) into one item
        # per sentence, using the translation's own punctuation as the boundary.
        if self._split_on_sentence and gen.output_text:
            tail = gen.output_text.rstrip()
            if tail and tail[-1] in _SENTENCE_END_CHARS:
                self._flush_segment()
                return

        # Otherwise (re)arm the silence timer on any real transcript / audio activity.
        # Guard on `.text` so empty keepalive transcription deltas don't keep an
        # utterance open indefinitely.
        has_activity = (
            (server_content.input_transcription and server_content.input_transcription.text)
            or (server_content.output_transcription and server_content.output_transcription.text)
            or server_content.model_turn
        )
        if has_activity:
            self._arm_segment_flush()

    def _start_new_generation(self) -> None:
        # A fresh generation starts a new utterance — drop any stale flush timer so
        # it can't finalize the new item prematurely.
        self._cancel_segment_flush()
        super()._start_new_generation()

    def _arm_segment_flush(self) -> None:
        self._cancel_segment_flush()
        self._segment_flush_handle = asyncio.get_event_loop().call_later(
            self._segment_flush_delay, self._flush_segment
        )

    def _cancel_segment_flush(self) -> None:
        if self._segment_flush_handle is not None:
            self._segment_flush_handle.cancel()
            self._segment_flush_handle = None

    def _flush_segment(self) -> None:
        """Finalize the current utterance as its own conversation item."""
        self._cancel_segment_flush()
        gen = self._current_generation
        if gen is None or gen._done:
            return
        # Mark completed so the base receive loop doesn't treat the next chunk as a
        # stale turn (which would trigger a reconnect), then close the item. The
        # next transcript delta starts a fresh generation via the base recv loop.
        self._generation_completed = True
        self._mark_current_generation_done()

    # ------------------------------------------------------------------
    # Override: translation is passive — no prompt / tools / manual replies
    # ------------------------------------------------------------------

    async def update_instructions(self, instructions: str) -> None:
        # Kept for parity with the RealtimeSession contract but never sent — the
        # translate model takes no system prompt.
        self._opts.instructions = instructions

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        pass  # translate model keeps no server-side chat context

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        pass  # translate model does not support tools

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass  # translate model is audio-only

    def generate_reply(
        self, **kwargs: Any
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
        fut.set_exception(
            llm.RealtimeError("generate_reply is not supported in Gemini translation mode")
        )
        return fut

    async def aclose(self) -> None:
        self._cancel_segment_flush()
        await super().aclose()
