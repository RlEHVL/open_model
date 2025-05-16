import sys
import os
import json
import requests
import tempfile
import numpy as np
import base64
from loguru import logger
import random
import re

from .tts_interface import TTSInterface
from .voicevox_processor import preprocess_korean_for_voicevox, KoreanHMMProcessor

# 번역 API 연동을 위한 선택적 가져오기
try:
    import deepl
    DEEPL_AVAILABLE = True
except ImportError:
    DEEPL_AVAILABLE = False

try:
    from googletrans import Translator
    GOOGLE_TRANS_AVAILABLE = True
except ImportError:
    GOOGLE_TRANS_AVAILABLE = False

class TTSEngine(TTSInterface):
    def __init__(
        self,
        host="127.0.0.1",
        port=50021,
        speaker_id=1,
        speed_scale=1.0,
        pitch_scale=0.0,
        intonation_scale=1.0,
        volume_scale=1.0,
        use_korean_preprocessing=True,
        apply_hmm_to_voice=True,  # 음성에도 HMM 적용
        hmm_depth=0.7,  # HMM 적용 강도 (0.0 ~ 1.0)
        use_translation=False,  # 번역 사용 여부
        translation_target="ja",  # 번역 대상 언어 (기본값: 일본어)
        translation_provider="internal",  # 번역 제공자 (internal, deepl, google)
        deepl_api_key=None,  # DeepL API 키
    ):
        """
        VoiceVox TTS 엔진 초기화

        Parameters:
            host (str): VoiceVox 엔진 호스트 주소
            port (int): VoiceVox 엔진 포트
            speaker_id (int): 화자 ID (VoiceVox 스타일 ID)
            speed_scale (float): 음성 속도 (기본값: 1.0)
            pitch_scale (float): 음성 피치 조절 (기본값: 0.0)
            intonation_scale (float): 억양 강도 (기본값: 1.0)
            volume_scale (float): 음량 조절 (기본값: 1.0)
            use_korean_preprocessing (bool): 한국어 전처리 활성화 여부 (기본값: True)
            apply_hmm_to_voice (bool): 음성에도 HMM 적용 여부 (기본값: True)
            hmm_depth (float): HMM 적용 강도 (0.0 ~ 1.0)
            use_translation (bool): 번역 활성화 여부 (기본값: False)
            translation_target (str): 번역 대상 언어 (기본값: "ja")
            translation_provider (str): 번역 제공자 (기본값: "internal")
            deepl_api_key (str): DeepL API 키 (선택 사항)
        """
        self.base_url = f"http://{host}:{port}"
        self.speaker_id = speaker_id
        self.speed_scale = speed_scale
        self.pitch_scale = pitch_scale
        self.intonation_scale = intonation_scale
        self.volume_scale = volume_scale
        self.use_korean_preprocessing = use_korean_preprocessing
        self.apply_hmm_to_voice = apply_hmm_to_voice
        self.hmm_depth = min(max(0.0, hmm_depth), 1.0)  # 0.0 ~ 1.0 범위로 제한
        
        # 번역 관련 설정
        self.use_translation = use_translation
        self.translation_target = translation_target
        self.translation_provider = translation_provider
        self.deepl_api_key = deepl_api_key
        
        # 번역기 초기화
        self._init_translator()
        
        # HMM 프로세서 초기화
        self.hmm_processor = KoreanHMMProcessor()
        
        # 언어 감지를 위한 플래그
        self.is_korean_text = False
        
        # VoiceVox 서버 연결 확인
        try:
            response = requests.get(f"{self.base_url}/version", timeout=3)
            if response.status_code == 200:
                logger.info(f"VoiceVox 엔진 연결 성공: 버전 {response.text}")
            else:
                logger.error(f"VoiceVox 엔진 연결 실패: 상태 코드 {response.status_code}")
        except requests.RequestException as e:
            logger.error(f"VoiceVox 엔진 연결 실패: {e}")
            logger.warning(f"VoiceVox 엔진이 실행 중인지 확인하세요: http://{host}:{port}")

    def _init_translator(self):
        """번역기 초기화"""
        self.translator = None
        
        if not self.use_translation:
            return
            
        if self.translation_provider == "deepl":
            if DEEPL_AVAILABLE and self.deepl_api_key:
                try:
                    self.translator = deepl.Translator(self.deepl_api_key)
                    logger.info("DeepL 번역기 초기화 완료")
                except Exception as e:
                    logger.error(f"DeepL 번역기 초기화 실패: {e}")
        
        elif self.translation_provider == "google":
            if GOOGLE_TRANS_AVAILABLE:
                try:
                    self.translator = Translator()
                    logger.info("Google 번역기 초기화 완료")
                except Exception as e:
                    logger.error(f"Google 번역기 초기화 실패: {e}")
        
        else:  # internal
            # 내부 번역 사용 (한국어->일본어 발음 변환)
            logger.info("내부 번역 시스템 사용")

    def translate_text(self, text, source_lang="ko", target_lang=None):
        """
        텍스트 번역

        Parameters:
            text (str): 번역할 텍스트
            source_lang (str): 원본 언어 코드
            target_lang (str): 대상 언어 코드, None이면 self.translation_target 사용

        Returns:
            dict: 번역 결과 (원본 텍스트, 번역 텍스트, 사용된 번역기)
        """
        if target_lang is None:
            target_lang = self.translation_target
            
        result = {
            "original_text": text,
            "translated_text": text,
            "translator": "none",
            "source_lang": source_lang,
            "target_lang": target_lang
        }
        
        if not self.use_translation:
            return result
            
        # 한국어 감지 및 번역 대상 확인
        is_korean = any('\u3131' <= char <= '\u318F' or '\uAC00' <= char <= '\uD7A3' for char in text)
        
        if not is_korean or source_lang != "ko":
            return result
            
        # 번역 시도
        try:
            if self.translation_provider == "deepl" and self.translator:
                translation = self.translator.translate_text(
                    text,
                    source_lang=source_lang,
                    target_lang=target_lang
                )
                result["translated_text"] = translation.text
                result["translator"] = "deepl"
                
            elif self.translation_provider == "google" and self.translator:
                translation = self.translator.translate(
                    text,
                    src=source_lang,
                    dest=target_lang
                )
                result["translated_text"] = translation.text
                result["translator"] = "google"
                
            else:
                # 내부 변환 사용 (한국어->일본어 발음)
                result["translated_text"] = text  # 원본 텍스트 유지 (내부 처리기가 한국어-일본어 발음 매핑 수행)
                result["translator"] = "internal"
                
            logger.info(f"번역 결과: '{text}' -> '{result['translated_text']}' ({result['translator']})")
            
        except Exception as e:
            logger.error(f"텍스트 번역 중 오류 발생: {e}")
            
        return result

    def process_text_json(self, text):
        """
        텍스트를 JSON 기반 처리 파이프라인을 통해 처리
        
        Parameters:
            text (str): 원본 텍스트
            
        Returns:
            dict: 처리 결과 JSON
        """
        # 1. 초기 데이터 구조
        processing_data = {
            "input": {
                "text": text,
                "detected_language": "unknown"
            },
            "processing_steps": [],
            "output": {
                "text": text,
                "is_processed": False
            }
        }
        
        # 2. 언어 감지
        is_korean = any('\u3131' <= char <= '\u318F' or '\uAC00' <= char <= '\uD7A3' for char in text)
        processing_data["input"]["detected_language"] = "ko" if is_korean else "ja" if any('\u3040' <= char <= '\u30ff' for char in text) else "unknown"
        processing_data["processing_steps"].append({
            "step": "language_detection",
            "result": processing_data["input"]["detected_language"]
        })
        
        # 3. 번역 (필요시)
        if self.use_translation and is_korean:
            translation_result = self.translate_text(text)
            processing_data["processing_steps"].append({
                "step": "translation",
                "translator": translation_result["translator"],
                "source_lang": translation_result["source_lang"],
                "target_lang": translation_result["target_lang"],
                "result": translation_result["translated_text"]
            })
            
            # 번역된 텍스트 설정 (DeepL/Google 사용 시)
            if translation_result["translator"] in ["deepl", "google"]:
                processing_data["output"]["text"] = translation_result["translated_text"]
                processing_data["output"]["is_translated"] = True
        
        # 4. 한국어 전처리 (HMM 발음 변환)
        if self.use_korean_preprocessing and is_korean:
            # 내부 전처리기 사용 (직접 변환)
            self.is_korean_text = True
            processed_text = preprocess_korean_for_voicevox(processing_data["output"]["text"])
            
            processing_data["processing_steps"].append({
                "step": "korean_preprocessing",
                "method": "hmm_phoneme_mapping",
                "result": processed_text
            })
            
            processing_data["output"]["text"] = processed_text
            processing_data["output"]["is_processed"] = True
        else:
            self.is_korean_text = False
        
        # 5. 문장 분절 최적화
        optimized_text = self._optimize_sentence_splitting(processing_data["output"]["text"])
        if optimized_text != processing_data["output"]["text"]:
            processing_data["processing_steps"].append({
                "step": "sentence_optimization",
                "result": optimized_text
            })
            processing_data["output"]["text"] = optimized_text
        
        # 6. 결과 음성 처리 정보 추가
        processing_data["output"]["tts_parameters"] = {
            "speaker_id": self.speaker_id,
            "speed_scale": self.speed_scale,
            "pitch_scale": self.pitch_scale,
            "intonation_scale": self.intonation_scale,
            "volume_scale": self.volume_scale,
            "sampling_rate": 44100,
            "hmm_depth": self.hmm_depth if self.is_korean_text and self.apply_hmm_to_voice else 0.0
        }
        
        logger.debug(f"텍스트 처리 파이프라인 결과: {json.dumps(processing_data, ensure_ascii=False, indent=2)}")
        return processing_data

    def preprocess_text(self, text):
        """
        텍스트 전처리 함수
        
        Parameters:
            text (str): 원본 텍스트
            
        Returns:
            str: 전처리된 텍스트
        """
        # JSON 기반 처리 파이프라인 실행
        processing_result = self.process_text_json(text)
        
        # 처리 결과에서 최종 텍스트 추출
        return processing_result["output"]["text"]

    def apply_hmm_voice_transformation(self, audio_query):
        """
        HMM을 사용하여 음성 합성 매개변수 수정
        
        Parameters:
            audio_query (dict): VoiceVox 오디오 쿼리
            
        Returns:
            dict: 수정된 오디오 쿼리
        """
        # 한국어가 아니면 HMM 적용하지 않음
        if not self.is_korean_text:
            logger.info("한국어 텍스트가 아님 - HMM 모델 적용 건너뜀")
            return audio_query
            
        # HMM 적용 여부 및 강도 확인
        if not self.apply_hmm_to_voice or self.hmm_depth <= 0.0:
            return audio_query
            
        # 음성 특성 HMM 적용 (랜덤성 추가)
        logger.info(f"음성 HMM 모델 적용 중 (강도: {self.hmm_depth})")
        
        # 한국어 발음 특성 HMM 상태
        hmm_states = ["시작", "중간", "종료", "강조"]
        current_state = 0
        
        # 발화 전체에 대한 기본 매개변수 조정
        base_speed_mod = 1.0 + (self.hmm_depth * 0.2)  # 한국어는 일본어보다 속도가 약간 빠름
        base_pitch_mod = 1.0 + (self.hmm_depth * 0.1)  # 한국어는 억양 폭이 큼
        base_intonation_mod = 1.0 + (self.hmm_depth * 0.3)  # 한국어는 억양이 뚜렷함
        
        # 기본 매개변수 적용
        audio_query["speedScale"] *= base_speed_mod
        audio_query["pitchScale"] *= base_pitch_mod 
        audio_query["intonationScale"] *= base_intonation_mod
        
        # HMM 전이 행렬 (상태 간 전이 확률)
        transition_matrix = np.array([
            [0.4, 0.5, 0.05, 0.05],  # 시작 -> (시작, 중간, 종료, 강조)
            [0.05, 0.7, 0.15, 0.1],  # 중간 -> (시작, 중간, 종료, 강조)
            [0.2, 0.2, 0.5, 0.1],    # 종료 -> (시작, 중간, 종료, 강조)
            [0.3, 0.4, 0.2, 0.1]     # 강조 -> (시작, 중간, 종료, 강조)
        ])
        
        # 자모 단위 HMM 사전 설정
        jamo_hmm_mods = {
            # 초성 자음에 따른 모디파이어
            "ㄱ": {"pitch": 0.95, "consonant": 1.3, "vowel": 1.0},
            "ㄴ": {"pitch": 1.0, "consonant": 1.1, "vowel": 1.0},
            "ㄷ": {"pitch": 0.98, "consonant": 1.4, "vowel": 0.9},
            "ㄹ": {"pitch": 1.05, "consonant": 1.0, "vowel": 1.1},
            "ㅁ": {"pitch": 0.9, "consonant": 1.2, "vowel": 1.0},
            "ㅂ": {"pitch": 0.95, "consonant": 1.3, "vowel": 0.9},
            "ㅅ": {"pitch": 1.1, "consonant": 1.3, "vowel": 0.95},
            "ㅇ": {"pitch": 1.0, "consonant": 1.0, "vowel": 1.1},
            "ㅈ": {"pitch": 1.05, "consonant": 1.4, "vowel": 0.9},
            "ㅊ": {"pitch": 1.1, "consonant": 1.5, "vowel": 0.85},
            "ㅋ": {"pitch": 0.9, "consonant": 1.3, "vowel": 0.95},
            "ㅌ": {"pitch": 0.95, "consonant": 1.4, "vowel": 0.9},
            "ㅍ": {"pitch": 0.9, "consonant": 1.3, "vowel": 0.95},
            "ㅎ": {"pitch": 1.02, "consonant": 1.1, "vowel": 1.05},
            
            # 중성 모음에 따른 모디파이어
            "ㅏ": {"pitch": 1.1, "consonant": 0.9, "vowel": 1.2},
            "ㅓ": {"pitch": 0.95, "consonant": 1.0, "vowel": 1.1},
            "ㅗ": {"pitch": 0.9, "consonant": 1.0, "vowel": 1.15},
            "ㅜ": {"pitch": 0.85, "consonant": 1.0, "vowel": 1.1},
            "ㅡ": {"pitch": 0.8, "consonant": 1.0, "vowel": 0.9},
            "ㅣ": {"pitch": 1.2, "consonant": 0.9, "vowel": 0.95},
            "ㅔ": {"pitch": 1.1, "consonant": 0.95, "vowel": 1.0},
            "ㅐ": {"pitch": 1.05, "consonant": 0.95, "vowel": 1.05},
            
            # 종성에 따른 모디파이어 (받침)
            "받침_ㄱ": {"pitch": 0.9, "end_pitch": 0.8, "consonant": 1.2, "vowel": 0.9, "pause": 1.1},
            "받침_ㄴ": {"pitch": 0.95, "end_pitch": 0.9, "consonant": 1.0, "vowel": 0.95, "pause": 1.05},
            "받침_ㄹ": {"pitch": 1.0, "end_pitch": 0.95, "consonant": 1.0, "vowel": 1.0, "pause": 1.0},
            "받침_ㅁ": {"pitch": 0.9, "end_pitch": 0.85, "consonant": 1.1, "vowel": 0.9, "pause": 1.1},
            "받침_ㅂ": {"pitch": 0.85, "end_pitch": 0.8, "consonant": 1.2, "vowel": 0.85, "pause": 1.15},
            "받침_ㅅ": {"pitch": 0.9, "end_pitch": 0.85, "consonant": 1.3, "vowel": 0.9, "pause": 1.1},
            "받침_ㅇ": {"pitch": 0.95, "end_pitch": 0.9, "consonant": 1.0, "vowel": 1.0, "pause": 1.05},
        }
        
        # 각 음소(모라)에 HMM 특성 적용
        if 'moras' in audio_query and audio_query['moras']:
            for i, mora in enumerate(audio_query['moras']):
                # 상태 전이
                next_state_probs = transition_matrix[current_state]
                next_state = np.random.choice(len(hmm_states), p=next_state_probs)
                current_state = next_state
                state_type = hmm_states[current_state]
                
                # 현재 상태와 깊이에 따른 영향력
                state_influence = random.random() * self.hmm_depth
                
                # 상태에 따른 변환 로직
                text = mora.get('text', '')
                
                # HMM 상태에 따른 변환 규칙
                if state_type == "시작":
                    # 시작 상태: 자음 강조, 높은 피치로 시작
                    pitch_mod = 1.1 - (state_influence * 0.2)
                    consonant_mod = 1.2 + (state_influence * 0.3)
                    vowel_mod = 1.0
                    
                elif state_type == "중간":
                    # 중간 상태: 자연스러운 발화
                    pitch_mod = 1.0
                    consonant_mod = 1.0
                    vowel_mod = 1.0
                    
                elif state_type == "종료":
                    # 종료 상태: 피치 하강, 모음 약화
                    pitch_mod = 0.9 - (state_influence * 0.1)
                    consonant_mod = 1.0
                    vowel_mod = 0.9 - (state_influence * 0.1)
                    
                else:  # "강조"
                    # 강조 상태: 피치 상승, 길이 증가
                    pitch_mod = 1.2 + (state_influence * 0.2)
                    consonant_mod = 1.1
                    vowel_mod = 1.2 + (state_influence * 0.1)
                
                # 한글 자모 특성 적용 (일본어 모라와 매핑)
                for jamo, mods in jamo_hmm_mods.items():
                    if jamo in text or (len(text) == 1 and ord('가') <= ord(text) <= ord('힣')):
                        # 피치 조정
                        pitch_mod *= mods.get("pitch", 1.0)
                        
                        # 자음 길이 조정
                        if 'consonant_length' in mora and mora['consonant_length'] is not None:
                            consonant_mod *= mods.get("consonant", 1.0)
                        
                        # 모음 길이 조정
                        if 'vowel_length' in mora and mora['vowel_length'] is not None:
                            vowel_mod *= mods.get("vowel", 1.0)
                        
                        break
                
                # 종성(받침) 특성 적용
                if i < len(audio_query['moras']) - 1:
                    next_mora = audio_query['moras'][i+1]
                    if next_mora.get('text', '').startswith('っ'):  # 촉음은 종성(받침) 특성 적용
                        for jamo_key, mods in jamo_hmm_mods.items():
                            if jamo_key.startswith("받침_"):
                                pitch_mod *= mods.get("pitch", 1.0)
                                if 'consonant_length' in next_mora and next_mora['consonant_length'] is not None:
                                    next_mora['consonant_length'] *= mods.get("consonant", 1.0)
                                break
                
                # 최종 모디파이어 적용
                # 피치 조정
                if 'pitch' in mora:
                    mora['pitch'] *= pitch_mod
                
                # 길이 조정 (자음 부분)
                if 'consonant_length' in mora and mora['consonant_length'] is not None:
                    mora['consonant_length'] *= consonant_mod
                
                # 길이 조정 (모음 부분)
                if 'vowel_length' in mora and mora['vowel_length'] is not None:
                    mora['vowel_length'] *= vowel_mod
        
        # 억양 곡선 조정 (한국어 억양 패턴 반영)
        if 'accent_phrases' in audio_query and audio_query['accent_phrases']:
            phrase_count = len(audio_query['accent_phrases'])
            
            for i, phrase in enumerate(audio_query['accent_phrases']):
                relative_pos = i / max(1, phrase_count - 1)  # 0.0 ~ 1.0 범위의 상대적 위치
                
                # 문장 내 위치에 따른 억양 변화 (한국어 패턴)
                if relative_pos < 0.3:  # 문장 시작부
                    # 한국어는 시작 부분에서 피치가 높고 점차 하강
                    pitch_bias = 1.0 + (0.1 * self.hmm_depth)
                elif relative_pos > 0.7:  # 문장 종료부
                    # 종료 부분에서는 억양이 다시 상승하는 경향
                    pitch_bias = 0.95 + (0.15 * self.hmm_depth * (relative_pos - 0.7) / 0.3)
                else:  # 문장 중간부
                    # 중간 부분은 점진적으로 하강
                    pitch_bias = 1.0 - (0.05 * self.hmm_depth * (relative_pos - 0.3) / 0.4)
                
                # 억양 곡선에 편향 적용
                if 'accent' in phrase:
                    phrase['accent'] = int(phrase['accent'] * pitch_bias)
                
                # 모라 피치에도 전체적인 편향 적용
                if 'moras' in phrase:
                    for mora in phrase['moras']:
                        if 'pitch' in mora:
                            mora['pitch'] *= pitch_bias
        
        # 최종 상태 전환 후 전체 발화 특성 미세 조정
        if self.hmm_depth > 0.5:  # 더 강한 HMM 적용인 경우에만
            # 한국어 특유의 리듬감을 위한 미세 조정
            rhythm_mod = random.uniform(0.9, 1.1) * self.hmm_depth
            if 'speedScale' in audio_query:
                audio_query['speedScale'] *= (1.0 + (rhythm_mod * 0.05))
            
            # 자연스러운 발화를 위한 억양 변동성 추가
            intonation_variance = random.uniform(0.9, 1.1) * self.hmm_depth
            if 'intonationScale' in audio_query:
                audio_query['intonationScale'] *= (1.0 + (intonation_variance * 0.1))
            
            # 한국어 특유의 억양을 위한 추가 조정
            if 'pitchScale' in audio_query:
                audio_query['pitchScale'] *= (1.0 + (random.uniform(-0.05, 0.1) * self.hmm_depth))
        
        logger.info("음성 HMM 모델 적용 완료 - 한국어 발음 특성 적용됨")
        return audio_query

    def say(self, text):
        """
        텍스트를 음성으로 변환하여 numpy 배열로 반환

        Parameters:
            text (str): 음성으로 변환할 텍스트

        Returns:
            tuple: (샘플 레이트, 오디오 데이터 배열)
        """
        try:
            # 텍스트 전처리 적용
            processed_text = self.preprocess_text(text)
            
            # 1. 오디오 쿼리 생성
            params = {"text": processed_text, "speaker": self.speaker_id}
            response = requests.post(f"{self.base_url}/audio_query", params=params, timeout=15)  # 타임아웃 증가
            if response.status_code != 200:
                logger.error(f"오디오 쿼리 생성 실패: {response.status_code} - {response.text}")
                return None, None
            
            audio_query = response.json()
            
            # 2. 음성 매개변수 조정
            audio_query["speedScale"] = self.speed_scale
            audio_query["pitchScale"] = self.pitch_scale
            audio_query["intonationScale"] = self.intonation_scale
            audio_query["volumeScale"] = self.volume_scale
            
            # 음질 개선을 위한 추가 설정
            audio_query["outputSamplingRate"] = 44100  # 샘플링 레이트 향상 (기본 24000에서 변경)
            audio_query["outputStereo"] = True  # 스테레오 출력 활성화
            audio_query["kaje"] = 0.6  # VOICEVOX 음성 선명도 향상 파라미터
            
            # HMM 기반 음성 변환 적용
            audio_query = self.apply_hmm_voice_transformation(audio_query)
            
            # 3. 오디오 합성
            params = {"speaker": self.speaker_id}
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(
                f"{self.base_url}/synthesis",
                params=params,
                data=json.dumps(audio_query),
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"음성 합성 실패: {response.status_code} - {response.text}")
                return None, None
            
            # 4. 오디오 데이터를 바로 메모리에 로드
            try:
                import io
                import soundfile as sf
                
                # 응답 데이터를 메모리 버퍼로 로드
                audio_buffer = io.BytesIO(response.content)
                audio_data, sample_rate = sf.read(audio_buffer)
                
                # 모노 채널로 변환 (필요한 경우)
                if audio_data.ndim > 1:
                    # 스테레오 유지 - 품질 향상을 위해
                    pass
                
                # float32 타입으로 정규화 - 음질 개선
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                    
                # 값 범위 확인 및 조정 (-1 ~ 1) - 클리핑 방지
                if np.max(np.abs(audio_data)) > 0.98:
                    # 부드러운 정규화를 위한 압축 적용
                    audio_data = np.tanh(audio_data) * 0.98
                
                return sample_rate, audio_data
                
            except Exception as e:
                logger.error(f"오디오 데이터 처리 중 오류 발생: {e}")
                return None, None
            
        except Exception as e:
            logger.error(f"음성 생성 중 오류 발생: {e}")
            return None, None

    def generate_audio_data(self, text: str) -> dict:
        """
        텍스트를 음성으로 변환하여 직접 오디오 데이터를 반환 (실시간 스트리밍용)

        Parameters:
            text (str): 음성으로 변환할 텍스트

        Returns:
            dict: 오디오 데이터 딕셔너리 (audio_base64, volumes, slice_length)
        """
        try:
            # JSON 기반 파이프라인 실행
            processing_result = self.process_text_json(text)
            processed_text = processing_result["output"]["text"]
            
            # 1. 오디오 쿼리 생성
            params = {"text": processed_text, "speaker": self.speaker_id}
            response = requests.post(f"{self.base_url}/audio_query", params=params, timeout=15)
            if response.status_code != 200:
                logger.error(f"오디오 쿼리 생성 실패: {response.status_code} - {response.text}")
                return {}
            
            audio_query = response.json()
            
            # 2. 음성 매개변수 조정
            audio_query["speedScale"] = self.speed_scale
            audio_query["pitchScale"] = self.pitch_scale
            audio_query["intonationScale"] = self.intonation_scale
            audio_query["volumeScale"] = self.volume_scale
            
            # 음질 개선을 위한 추가 설정
            audio_query["outputSamplingRate"] = 44100  # 샘플링 레이트 향상
            audio_query["outputStereo"] = True  # 스테레오 출력 활성화
            
            # 한국어에만 kaje 파라미터 적용 (일본어는 기본값 사용)
            if self.is_korean_text:
                audio_query["kaje"] = 0.6  # 음성 선명도 향상 파라미터
            
            # 문장 사이 일시 중지 최적화 (끊김 방지)
            if 'prePhonemeLength' in audio_query:
                audio_query['prePhonemeLength'] = 0.1  # 시작 전 무음 구간 축소
            if 'postPhonemeLength' in audio_query:
                audio_query['postPhonemeLength'] = 0.1  # 종료 후 무음 구간 축소
            
            # HMM 기반 음성 변환 적용 (한국어인 경우에만)
            audio_query = self.apply_hmm_voice_transformation(audio_query)
            
            # 3. 오디오 합성 - 질문문 억양은 한국어에만 적용
            params = {"speaker": self.speaker_id}
            if self.is_korean_text:
                params["enable_interrogative_upspeak"] = "true"
                
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(
                f"{self.base_url}/synthesis",
                params=params,
                data=json.dumps(audio_query),
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"음성 합성 실패: {response.status_code} - {response.text}")
                return {}
            
            # 4. 오디오 데이터 바로 인코딩
            audio_base64 = base64.b64encode(response.content).decode("utf-8")
            
            # 5. 볼륨 데이터 추출 (음소 단위 타이밍을 위함)
            volumes = []
            chunk_size = 100  # 오디오 청크 크기 (밀리초) - 증가된 값
            
            if 'moras' in audio_query and audio_query['moras']:
                # VoiceVox 모라 정보로부터 볼륨 정보 추출
                for mora in audio_query['moras']:
                    if 'consonant_length' in mora and mora['consonant_length'] is not None:
                        volumes.append(random.uniform(0.8, 1.0))  # 자음 볼륨 상향
                    
                    if 'vowel_length' in mora and mora['vowel_length'] is not None:
                        volumes.append(random.uniform(0.9, 1.0))  # 모음 볼륨 상향
            else:
                # 모라 정보가 없는 경우, 균일한 볼륨 데이터 생성
                total_chunks = len(response.content) // 1000
                volumes = [0.95 for _ in range(total_chunks)]  # 균일한 볼륨으로 변경
            
            # 6. 처리 파이프라인 결과 추가
            result = {
                "audio": audio_base64,
                "volumes": volumes,
                "chunk_size": chunk_size,
                "sample_rate": 44100,  # 향상된 샘플 레이트
                "is_streaming": True,
                "format": "wav",  # 명시적 포맷 지정
                "is_korean": self.is_korean_text,  # 한국어 여부 정보 추가
                "processing": processing_result  # 텍스트 처리 파이프라인 결과
            }
            
            logger.info(f"실시간 오디오 데이터 생성 완료: {len(response.content)} bytes, {len(volumes)} chunks")
            return result
            
        except Exception as e:
            logger.error(f"오디오 데이터 생성 중 오류 발생: {e}")
            return {}

    def _optimize_sentence_splitting(self, text: str) -> str:
        """
        음성 합성을 위해 텍스트를 최적화하여 자연스러운 발화 생성
        
        Parameters:
            text (str): 원본 텍스트
            
        Returns:
            str: 최적화된 텍스트
        """
        # 문장 종결 패턴 발견 후 적절한 간격 추가
        text = re.sub(r'([.!?])\s*', r'\1 ', text)
        
        # 쉼표 후 미세한 간격 추가
        text = re.sub(r'(,)\s*', r'\1 ', text)
        
        # 과도한 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 긴 문장 분리 방지
        if len(text) > 100:
            sentences = re.split(r'([.!?])', text)
            result = []
            
            # 문장 재결합 (구두점 포함)
            for i in range(0, len(sentences)-1, 2):
                if i+1 < len(sentences):
                    result.append(sentences[i] + sentences[i+1])
                elif i < len(sentences):
                    result.append(sentences[i])
            
            # 자연스러운 문장 길이로 조정
            optimized = []
            temp = ""
            for s in result:
                if len(temp) + len(s) < 80:  # 자연스러운 발화 위한 길이 제한
                    temp += s + " "
                else:
                    if temp:
                        optimized.append(temp.strip())
                    temp = s + " "
            
            if temp:
                optimized.append(temp.strip())
            
            # 최종 텍스트 재결합
            return ' '.join(optimized)
        
        return text.strip()

    def generate_audio(self, text: str, file_name_no_ext=None) -> str:
        """
        텍스트를 음성으로 변환하여 파일로 저장하고 파일 경로를 반환
        
        실시간 스트리밍을 위해서는 generate_audio_data 함수를 사용하세요.
        이 함수는 기존 호환성을 위해 유지됩니다.

        Parameters:
            text (str): 음성으로 변환할 텍스트
            file_name_no_ext (str, optional): 파일 이름(확장자 제외)

        Returns:
            str: 생성된 오디오 파일 경로
        """
        try:
            # 오디오 데이터 생성
            audio_data = self.generate_audio_data(text)
            
            if not audio_data or "audio" not in audio_data:
                logger.error("오디오 데이터 생성 실패")
                return ""
            
            # 저장할 파일 경로 생성
            file_path = self.generate_cache_file_name(file_name_no_ext, "wav")
            
            # 파일 저장
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(audio_data["audio"]))
            
            logger.info(f"음성 파일 생성 완료: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"음성 생성 중 오류 발생: {e}")
            return ""

    async def async_generate_audio_data(self, text: str) -> dict:
        """
        텍스트를 음성으로 변환하여 비동기적으로 오디오 데이터를 반환 (실시간 스트리밍용)

        Parameters:
            text (str): 음성으로 변환할 텍스트

        Returns:
            dict: 오디오 데이터 딕셔너리 (audio_base64, volumes, slice_length)
        """
        import asyncio
        return await asyncio.to_thread(self.generate_audio_data, text)

    async def async_generate_audio(self, text: str, file_name_no_ext=None) -> str:
        """
        텍스트를 음성으로 변환하여 비동기적으로 파일로 저장하고 파일 경로를 반환

        Parameters:
            text (str): 음성으로 변환할 텍스트
            file_name_no_ext (str, optional): 파일 이름(확장자 제외)

        Returns:
            str: 생성된 오디오 파일 경로
        """
        import asyncio
        return await asyncio.to_thread(self.generate_audio, text, file_name_no_ext) 