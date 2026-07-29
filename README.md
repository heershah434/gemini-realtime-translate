# livekit-plugins-gemini-translate

LiveKit Agents plugin for **Google Gemini Live realtime translation**
(`gemini-3.5-live-translate-preview`).

It provides a `GeminiTranslationModel` / `GeminiTranslationSession` pair that
performs speech-to-speech translation (ASR → translate → TTS) over the Gemini
Live API, emitting translated audio plus the source and translated transcripts.
It is a drop-in sibling of the OpenAI realtime-translate and Azure Speech
translate plugins and plugs into the same `RealtimeTranslator` pipeline.

## How it works

The plugin **subclasses** `livekit-plugins-google`'s realtime `RealtimeModel` /
`RealtimeSession` rather than re-implementing the Live API. Gemini Live Translate
reuses the same `server_content` wire format as the regular realtime model, so the
base session already decodes translated audio (`model_turn.parts[].inline_data`),
the source transcript (`input_transcription`) and the translated transcript
(`output_transcription`). The only override is the connect config: point the
session at the translate model and inject a `translation_config`.

## Installation

```bash
pip install livekit-plugins-gemini-translate
```

## Usage

```python
from livekit.plugins.gemini_translate import GeminiTranslationModel

model = GeminiTranslationModel(
    api_key="YOUR_GOOGLE_API_KEY",   # or set GOOGLE_API_KEY
    target_language="hi",            # BCP-47 target language code
    echo_target_language=True,       # echo speech already in the target language
)
```

VertexAI is also supported by forwarding `vertexai=True`, `project`, `location`
and `credentials` to the constructor.

## Configuration

| Argument | Default | Description |
|---|---|---|
| `target_language` | — (required) | BCP-47 code for the spoken translation output. |
| `echo_target_language` | `True` | Echo speech already in the target language verbatim (`True`) or stay silent (`False`). |
| `model` | `gemini-3.5-live-translate-preview` | Gemini Live translate model name. |
| `api_key` | `GOOGLE_API_KEY` env | Google Gemini API key (omit for VertexAI). |
| `input_audio_transcription` | enabled | Source-language transcript config. |
| `output_audio_transcription` | enabled | Translated transcript config. |
| `realtime_input_config` | server default | Full `types.RealtimeInputConfig` passthrough (server-side VAD / turn detection). Wins over the `vad_*` args below. |
| `vad_silence_duration_ms` | server default | End-of-speech silence (ms) before the model commits the turn. **Biggest latency lever** — lower is snappier. |
| `vad_end_sensitivity` | server default | `"HIGH"` detects end-of-speech faster (lower latency); `"LOW"` waits longer. |
| `vad_start_sensitivity` | server default | `"HIGH"` / `"LOW"` — eagerness to detect speech start. |
| `vad_prefix_padding_ms` | server default | Audio kept before detected speech start (ms). |
| `context_window_compression` | off | Optional `types.ContextWindowCompressionConfig` passthrough. |
| `segment_flush_delay` | `0.6` | Silence (s) before a transcript **item** is finalized. Does **not** delay audio. |
| `split_on_sentence` | `True` | Finalize a transcript item on each sentence-ending punctuation. |

### Tuning for lowest latency

The translated **audio** and live transcript already stream frame-by-frame as they
arrive — the main tunable delay is how long Gemini's server-side VAD waits after the
speaker stops before it commits the turn and produces the translation. Shrink it:

```python
model = GeminiTranslationModel(
    api_key="YOUR_GOOGLE_API_KEY",
    target_language="hi",
    echo_target_language=False,       # avoid re-voicing target-language audio (feedback)
    vad_silence_duration_ms=150,      # commit the turn ~fast (default is much higher)
    vad_end_sensitivity="HIGH",       # detect end-of-speech sooner
    vad_prefix_padding_ms=20,
    segment_flush_delay=0.4,          # snappier transcript finalization
)
```

Trade-off: very low `vad_silence_duration_ms` (< ~100 ms) can clip speakers who
pause mid-thought and split one sentence into two. 150–300 ms is a good starting
range; measure against your audio and back off if you see clipping.

## Requirements

- `livekit-agents>=1.5.4`
- `livekit-plugins-google>=1.5.4`
- `google-genai>=1.55` (for `types.TranslationConfig`)

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## References

- [Gemini Live Translate docs](https://ai.google.dev/gemini-api/docs/live-api/live-translate)
