import concurrent.futures
import copy
from collections.abc import Generator
from functools import reduce
from io import BytesIO
from typing import Any, Optional

from pydub import AudioSegment

from dify_plugin.entities.model import AIModelEntity
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeBadRequestError,
)
from dify_plugin.interfaces.model.tts_model import TTSModel
from ..common import _CommonAzureOpenAI
from ..constants import TTS_BASE_MODELS, AzureBaseModel


class AzureOpenAIText2SpeechModel(_CommonAzureOpenAI, TTSModel):
    """
    Model class for OpenAI Speech to text model.
    """

    def _invoke(
        self,
        model: str,
        tenant_id: str,
        credentials: dict,
        content_text: str,
        voice: str,
        user: Optional[str] = None,
    ) -> Any:
        """
        _invoke text2speech model

        :param model: model name
        :param tenant_id: user tenant id
        :param credentials: model credentials
        :param content_text: text content to be translated
        :param voice: model timbre
        :param user: unique user id
        :return: text translated to audio file
        """
        if not voice or voice not in [
            d["value"]
            for d in self.get_tts_model_voices(model=model, credentials=credentials)
        ]:
            voice = self._get_model_default_voice(model, credentials)
        return self._tts_invoke_streaming(
            model=model, credentials=credentials, content_text=content_text, voice=voice
        )

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        validate credentials text2speech model

        :param model: model name
        :param credentials: model credentials
        :return: text translated to audio file
        """
        try:
            self._tts_invoke_streaming(
                model=model,
                credentials=credentials,
                content_text="Hello Dify!",
                voice=self._get_model_default_voice(model, credentials),
            )
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    def _tts_invoke(
        self, model: str, credentials: dict, content_text: str, voice: str
    ) -> bytes:
        """
        Non-streaming TTS invoke. Splits text into sentences, fetches audio
        for each in parallel, then combines segments with pydub to produce a
        correctly formatted audio file.

        :param model: model name
        :param credentials: model credentials
        :param content_text: text content to be translated
        :param voice: model timbre
        :return: combined audio bytes
        """
        audio_type = self._get_model_audio_type(model, credentials)
        word_limit = self._get_model_word_limit(model, credentials) or 500
        max_workers = self._get_model_workers_limit(model, credentials)

        try:
            sentences = list(
                self._split_text_into_sentences(
                    org_text=content_text, max_length=word_limit
                )
            )
            audio_bytes_list: list[bytes] = []

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = [
                    executor.submit(
                        self._process_sentence,
                        sentence=sentence,
                        model=model,
                        voice=voice,
                        credentials=credentials,
                    )
                    for sentence in sentences
                ]
                for future in futures:
                    try:
                        result = future.result()
                        if result:
                            audio_bytes_list.append(result)
                    except Exception as ex:
                        raise InvokeBadRequestError(str(ex))

            if not audio_bytes_list:
                raise InvokeBadRequestError("No audio bytes found")

            # Combine segments with pydub, then re-export in target format.
            audio_segments = [
                AudioSegment.from_file(BytesIO(buf), format=audio_type)
                for buf in audio_bytes_list
                if buf
            ]
            combined = reduce(lambda a, b: a + b, audio_segments)
            buffer = BytesIO()
            combined.export(buffer, format=audio_type)
            buffer.seek(0)
            return buffer.read()

        except InvokeBadRequestError:
            raise
        except Exception as ex:
            raise InvokeBadRequestError(str(ex))

    def _tts_invoke_streaming(
        self, model: str, credentials: dict, content_text: str, voice: str
    ) -> Generator[bytes, None, None]:
        """
        Streaming TTS invoke.

        :param model: model name
        :param credentials: model credentials
        :param content_text: text content to be translated
        :param voice: model timbre
        :return: generator of audio byte chunks
        """
        try:
            audio_type = self._get_model_audio_type(model, credentials)
            client = self._create_client(credentials)
            max_length = 3500
            if len(content_text) > max_length:
                sentences = self._split_text_into_sentences(
                    content_text, max_length=max_length
                )
                executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(3, len(sentences))
                )
                futures = [
                    executor.submit(
                        client.audio.speech.with_streaming_response.create,
                        model=model,
                        response_format=audio_type,
                        input=sentences[i],
                        voice=voice,
                    )
                    for i in range(len(sentences))
                ]
                for future in futures:
                    yield from future.result().__enter__().iter_bytes(1024)
            else:
                response = client.audio.speech.with_streaming_response.create(
                    model=model,
                    voice=voice,
                    response_format=audio_type,
                    input=content_text.strip(),
                )
                yield from response.__enter__().iter_bytes(1024)
        except Exception as ex:
            raise InvokeBadRequestError(str(ex))

    def _process_sentence(self, sentence: str, model: str, voice, credentials: dict):
        """
        Invoke Azure OpenAI TTS API for a single sentence.

        :param model: model name
        :param credentials: model credentials
        :param voice: model timbre
        :param sentence: text content to be translated
        :return: audio bytes
        """
        audio_type = self._get_model_audio_type(model, credentials)
        client = self._create_client(credentials)
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=sentence.strip(),
            response_format=audio_type,
        )
        if isinstance(response.read(), bytes):
            return response.read()

    def get_customizable_model_schema(
        self, model: str, credentials: dict
    ) -> Optional[AIModelEntity]:
        base_model_name = self._get_base_model_name(credentials)
        ai_model_entity = self._get_ai_model_entity(base_model_name, model)
        return ai_model_entity.entity if ai_model_entity else None

    @staticmethod
    def _get_ai_model_entity(base_model_name: str, model: str) -> AzureBaseModel | None:
        for ai_model_entity in TTS_BASE_MODELS:
            if ai_model_entity.base_model_name == base_model_name:
                ai_model_entity_copy = copy.deepcopy(ai_model_entity)
                ai_model_entity_copy.entity.model = model
                ai_model_entity_copy.entity.label.en_US = model
                ai_model_entity_copy.entity.label.zh_Hans = model
                return ai_model_entity_copy
        return None
