import sys
import types
import unittest
from unittest.mock import Mock, MagicMock

# ---------------------------------------------------------------------------
# Test stubs – mock heavy / unavailable deps so we can import tts.py purely
# ---------------------------------------------------------------------------

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# pydub mock (requires audioop, removed in 3.13+)
sys.modules.setdefault("pydub", MagicMock())
sys.modules.setdefault("pydub.AudioSegment", MagicMock())


def _install_stubs():
    if "httpx" not in sys.modules:
        m = types.ModuleType("httpx")
        m.Timeout = type("Timeout", (), {"__init__": lambda s, *a, **k: None})
        sys.modules["httpx"] = m

    if "openai" not in sys.modules:
        m = types.ModuleType("openai")
        for name in (
            "OpenAI",
            "AzureOpenAI",
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "RateLimitError",
            "AuthenticationError",
            "PermissionDeniedError",
            "BadRequestError",
            "NotFoundError",
            "UnprocessableEntityError",
            "APIError",
        ):
            setattr(m, name, type(name, (Exception,), {}))
        sys.modules["openai"] = m

    for mod_path, attrs in {
        "dify_plugin.entities.model": {"AIModelEntity": type("AIModelEntity", (), {})},
        "dify_plugin.errors.model": {
            "CredentialsValidateFailedError": type("CredErr", (Exception,), {}),
            "InvokeBadRequestError": type("InvErr", (Exception,), {}),
        },
        "dify_plugin.interfaces.model.tts_model": {
            "TTSModel": type("TTSModel", (), {}),
        },
    }.items():
        if mod_path not in sys.modules:
            m = types.ModuleType(mod_path)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[mod_path] = m

    if "models.constants" not in sys.modules:
        m = types.ModuleType("models.constants")
        m.AzureBaseModel = type("AzureBaseModel", (), {})
        m.TTS_BASE_MODELS = []
        sys.modules["models.constants"] = m


_install_stubs()

from models.tts.tts import AzureOpenAIText2SpeechModel  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(audio_type="mp3"):
    """Create a model instance with sensible defaults for testing."""
    model = AzureOpenAIText2SpeechModel.__new__(AzureOpenAIText2SpeechModel)
    model._get_model_audio_type = Mock(return_value=audio_type)
    model._get_model_word_limit = Mock(return_value=500)
    model._get_model_workers_limit = Mock(return_value=3)
    return model


class _StreamResp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_bytes(self, n):
        yield b"ch1"
        yield b"ch2"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStreamingUsesAudioType(unittest.TestCase):
    """Streaming path should pass audio_type as response_format."""

    def test_streaming_uses_configured_format(self):
        captured = {}

        def create(**kw):
            captured.update(kw)
            return _StreamResp()

        model = _make_model(audio_type="wav")
        model._create_client = Mock(
            return_value=types.SimpleNamespace(
                audio=types.SimpleNamespace(
                    speech=types.SimpleNamespace(
                        with_streaming_response=types.SimpleNamespace(create=create)
                    )
                )
            )
        )

        list(
            model._tts_invoke_streaming(
                model="tts-1", credentials={}, content_text="hi", voice="alloy"
            )
        )
        self.assertEqual(captured["response_format"], "wav")

    def test_streaming_long_text(self):
        calls = []

        def create(**kw):
            calls.append(kw["response_format"])
            return _StreamResp()

        model = _make_model(audio_type="opus")
        model._create_client = Mock(
            return_value=types.SimpleNamespace(
                audio=types.SimpleNamespace(
                    speech=types.SimpleNamespace(
                        with_streaming_response=types.SimpleNamespace(create=create)
                    )
                )
            )
        )

        text = "x" * 4000
        list(
            model._tts_invoke_streaming(
                model="tts-1", credentials={}, content_text=text, voice="alloy"
            )
        )
        self.assertTrue(all(f == "opus" for f in calls))


class TestProcessSentencePassesAudioType(unittest.TestCase):
    """_process_sentence should send audio_type as response_format."""

    def test_process_uses_audio_type(self):
        captured = {}
        mock_resp = Mock()
        mock_resp.read.return_value = b"data"

        def create(**kw):
            captured.update(kw)
            return mock_resp

        model = _make_model(audio_type="flac")
        model._create_client = Mock(
            return_value=types.SimpleNamespace(
                audio=types.SimpleNamespace(
                    speech=types.SimpleNamespace(create=create)
                )
            )
        )

        result = model._process_sentence("hello", "tts-1", "alloy", {})
        self.assertEqual(result, b"data")
        self.assertEqual(captured["response_format"], "flac")


class TestNonStreamingInvoke(unittest.TestCase):
    """_tts_invoke should combine segments via pydub."""

    def test_invokes_pydub_combine(self):
        model = _make_model(audio_type="wav")
        model._split_text_into_sentences = Mock(return_value=["hello", "world"])
        model._process_sentence = Mock(side_effect=lambda **kw: b"seg-bytes")

        # Mock pydub AudioSegment at module level
        import models.tts.tts as tts_mod

        mock_seg1 = Mock()
        mock_seg2 = Mock()
        mock_combined = Mock()
        mock_seg1.__add__ = Mock(return_value=mock_combined)
        mock_combined.export = Mock()

        with unittest.mock.patch.object(tts_mod, "AudioSegment", create=False) as mock_as:
            mock_as.from_file.side_effect = [mock_seg1, mock_seg2]
            result = model._tts_invoke("tts-1", {}, "hello world", "alloy")

        mock_combined.export.assert_called_once()


if __name__ == "__main__":
    unittest.main()
