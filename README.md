# Open-LLM 사용 가이드

이 문서는 Open-llm 프로젝트의 설치 및 사용 방법을 설명합니다.

conf.yaml에서 api는 자신의 키를 입력하세요

## 후원하기
이 프로젝트가 도움이 되셨다면 개발자에게 후원을 고려해보세요:

[![후원하기](https://img.shields.io/badge/후원하기-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/rlehvl)
[![GitHub 스폰서](https://img.shields.io/badge/GitHub_스폰서-ea4aaa?style=for-the-badge&logo=github-sponsors&logoColor=white)](https://github.com/sponsors/RlEHVL)

## 목차
- [설치 방법](#설치-방법)
- [기본 설정](#기본-설정)
- [실행 방법](#실행-방법)
- [주요 기능](#주요-기능)
- [문제 해결](#문제-해결)

## 설치 방법

### 필수 요구 사항
- Python 3.10 이상
- Git
- CUDA 지원 GPU (선택 사항이지만 권장)

### 설치 단계

1. 레포지토리 클론:
```bash
git clone https://github.com/RlEHVL/open_llm.git
cd open_llm
```

2. 가상 환경 생성 및 활성화:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

3. 필요한 패키지 설치:
```bash
pip install -e .
```

## 기본 설정

1. `conf.yaml` 파일에서 기본 설정을 확인합니다. 이 파일에는 다음과 같은 주요 섹션이 있습니다:
   - `system_config`: 서버 설정
   - `character_config`: 캐릭터 설정
   - `asr_config`: 음성 인식 설정
   - `tts_config`: 음성 합성 설정

2. TTS(Text-to-Speech) 설정:
   - 기본적으로 VoiceVox를 사용합니다. [VoiceVox 엔진](https://github.com/VOICEVOX/voicevox_engine)을 설치하세요.
   - 설치 경로-> [![설치](https://voicevox.hiroshiba.jp/)
   - 또는 `tts_config` 섹션에서 다른 TTS 엔진(edge_tts, azure_tts 등)을 선택할 수 있습니다.

3. ASR(Automatic Speech Recognition) 설정:
   - 기본적으로 Azure Speech Recognition을 사용합니다.
   - 또는 `asr_config` 섹션에서 다른 ASR 엔진(faster_whisper, whisper_cpp 등)을 선택할 수 있습니다.

4. LLM(Large Language Model) 설정:
   - 기본적으로 Ollama를 사용합니다. [Ollama](https://ollama.ai/)를 설치하세요.
   - 또는 `llm_configs` 섹션에서 다른 LLM 제공자(OpenAI, Claude 등)를 설정할 수 있습니다.

## 실행 방법

1. 서버 실행:
```bash
python run_server.py
```

2. 웹 브라우저에서 접속:
   - 기본 주소: `http://localhost:12393`
   - 다른 기기에서 접속하려면 `conf.yaml`의 `host`를 `0.0.0.0`으로 설정하세요.

## 주요 기능

### 음성 상호작용
1. 마이크 아이콘을 클릭하고 말하면 AI가 응답합니다.
2. 음성 인식이 자동으로 실행되고 AI가 응답을 생성합니다.
3. 텍스트를 음성으로 변환하여 Live2D 모델이 립싱크와 함께 응답합니다.

### 텍스트 채팅
1. 텍스트 입력창에 메시지를 입력하고 전송할 수 있습니다.
2. AI는 텍스트와 음성으로 응답합니다.

### Live2D 모델 제어
1. 모델은 대화 내용에 따라 표정을 바꿉니다.
2. 모델 교체는 `model_dict.json` 파일과 `conf.yaml`에서 설정할 수 있습니다.

## 문제 해결

### 음성 인식 문제
- 마이크가 제대로 연결되어 있는지 확인하세요.
- 브라우저에서 마이크 권한이 허용되어 있는지 확인하세요.
- ASR 모델 설정이 올바른지 확인하세요.

### 음성 합성 문제
- VoiceVox 서버가 실행 중인지 확인하세요.
- 다른 TTS 엔진으로 전환해보세요.

### LLM 연결 문제
- Ollama가 실행 중인지 확인하세요.
- API 키가 올바르게 설정되어 있는지 확인하세요.
- 인터넷 연결을 확인하세요.

### 기타 문제
- 로그 폴더에서 로그 파일을 확인하여 오류를 진단할 수 있습니다.
- 이슈는 GitHub 레포지토리에 제출해주세요. 
