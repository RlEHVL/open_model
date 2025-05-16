# 일반 TTS 인터페이스 정의
from .tts_interface import TTSInterface
from .tts_factory import TTSFactory

# 명시적으로 voicevox_tts 모듈 추가
try:
    from .voicevox_tts import TTSEngine as VoiceVoxTTSEngine
except ImportError:
    pass
