from typing import Dict, List, Any, Optional
import re
import numpy as np
import random

class KoreanHMMProcessor:
    """한국어 텍스트를 VoiceVox에 적합한 형태로 변환하는 강화된 HMM 기반 프로세서"""
    
    def __init__(self):
        # 한국어 음소와 일본어 음소 간 매핑 (확장된 매핑)
        self.ko_to_ja_phoneme_map = {
            # 모음 (개선된 매핑)
            'ㅏ': 'あ', 'ㅓ': 'お', 'ㅗ': 'お', 'ㅜ': 'う', 
            'ㅡ': 'う', 'ㅣ': 'い', 'ㅐ': 'え', 'ㅔ': 'え',
            'ㅑ': 'や', 'ㅕ': 'よ', 'ㅛ': 'よ', 'ㅠ': 'ゆ',
            'ㅒ': 'いぇ', 'ㅖ': 'いぇ', 'ㅘ': 'わ', 'ㅙ': 'うぇ',
            'ㅚ': 'え', 'ㅝ': 'を', 'ㅞ': 'うぇ', 'ㅟ': 'うぃ',
            'ㅢ': 'うぃ',
            
            # 자음 (개선된 매핑)
            'ㄱ': 'か', 'ㄲ': 'っか', 'ㄴ': 'な', 'ㄷ': 'た', 
            'ㄸ': 'った', 'ㄹ': 'ら', 'ㅁ': 'ま', 'ㅂ': 'は', 
            'ㅃ': 'っぱ', 'ㅅ': 'さ', 'ㅆ': 'っさ', 'ㅇ': '', 
            'ㅈ': 'じゃ', 'ㅉ': 'っじゃ', 'ㅊ': 'ちゃ', 'ㅋ': 'か',
            'ㅌ': 'た', 'ㅍ': 'ぱ', 'ㅎ': 'は',
            
            # 종성 매핑 (강화)
            'ㄱ_종': 'く', 'ㄲ_종': 'っく', 'ㄳ_종': 'くす', 'ㄴ_종': 'ん',
            'ㄵ_종': 'んじ', 'ㄶ_종': 'んひ', 'ㄷ_종': 'と', 'ㄹ_종': 'る',
            'ㄺ_종': 'るく', 'ㄻ_종': 'るむ', 'ㄼ_종': 'るぷ', 'ㄽ_종': 'るす',
            'ㄾ_종': 'るて', 'ㄿ_종': 'るふ', 'ㅀ_종': 'るひ', 'ㅁ_종': 'む',
            'ㅂ_종': 'ぷ', 'ㅄ_종': 'ぷす', 'ㅅ_종': 'す', 'ㅆ_종': 'っす',
            'ㅇ_종': 'ん', 'ㅈ_종': 'じ', 'ㅊ_종': 'ち', 'ㅋ_종': 'く',
            'ㅌ_종': 'と', 'ㅍ_종': 'ふ', 'ㅎ_종': 'ひ',
            
            # 자주 사용되는 음절 조합 (추가)
            '가': 'か', '나': 'な', '다': 'た', '라': 'ら', '마': 'ま',
            '바': 'ば', '사': 'さ', '아': 'あ', '자': 'じゃ', '차': 'ちゃ',
            '카': 'か', '타': 'た', '파': 'ぱ', '하': 'は',
            '고': 'こ', '노': 'の', '도': 'ど', '로': 'ろ', '모': 'も',
            '보': 'ぼ', '소': 'そ', '오': 'お', '조': 'じょ', '초': 'ちょ',
            '코': 'こ', '토': 'と', '포': 'ぽ', '호': 'ほ',
            '구': 'く', '누': 'ぬ', '두': 'どぅ', '루': 'る', '무': 'む',
            '부': 'ぶ', '수': 'す', '우': 'う', '주': 'じゅ', '추': 'ちゅ',
            '쿠': 'く', '투': 'とぅ', '푸': 'ぷ', '후': 'ふ',
            '기': 'き', '니': 'に', '디': 'でぃ', '리': 'り', '미': 'み',
            '비': 'び', '시': 'し', '이': 'い', '지': 'じ', '치': 'ち',
            '키': 'き', '티': 'てぃ', '피': 'ぴ', '히': 'ひ',
            
            # 자주 사용되는 어미/조사
            '은': 'うん', '는': 'ぬん', '이': 'い', '가': 'か',
            '을': 'うる', '를': 'るる', '에': 'え', '의': 'え',
            '과': 'くゎ', '와': 'わ', '도': 'ど', '만': 'まん',
            '에서': 'えそ', '부터': 'ぷと', '까지': 'かじ',
            
            # 특수 단어 (가장 자주 사용되는 단어들)
            '안녕하세요': 'あんにょんはせよ',
            '감사합니다': 'かむさはむにだ',
            '미안합니다': 'みあんはむにだ',
            '사랑합니다': 'さらんはむにだ',
            '반갑습니다': 'ばんがっすむにだ',
            '잘 부탁드립니다': 'じゃる ぶたくどぅりむにだ',
            '처음 뵙겠습니다': 'ちょうむ べっけっすむにだ',
            '잘 지내세요': 'じゃる じねせよ',
            '어서오세요': 'おそおせよ',
            '질문이 있습니다': 'じるむに いっすむにだ',
            '네': 'ね',
            '아니요': 'あによ',
            '괜찮습니다': 'けんちゃなすむにだ',
            '좋아요': 'じょあよ',
            '잘했어요': 'じゃれっそよ'
        }
        
        # 한국어 보조 HMM 상태 (시작, 중간, 종료 패턴에 따른 발음 변화)
        self.ko_state_patterns = {
            "시작": {
                'ㄱ': 'か', 'ㄲ': 'っか', 'ㄷ': 'た', 
                'ㄸ': 'った', 'ㅂ': 'ぱ', 'ㅃ': 'っぱ', 
                'ㅅ': 'さ', 'ㅆ': 'っさ', 'ㅈ': 'じゃ', 
                'ㅉ': 'っじゃ', 'ㅊ': 'ちゃ', 'ㅋ': 'か',
                'ㅌ': 'た', 'ㅍ': 'ぱ'
            },
            "중간": {
                'ㄱ': 'が', 'ㄲ': 'っか', 'ㄷ': 'だ', 
                'ㄸ': 'った', 'ㅂ': 'ば', 'ㅃ': 'っぱ', 
                'ㅅ': 'さ', 'ㅆ': 'っさ', 'ㅈ': 'じゃ', 
                'ㅉ': 'っじゃ', 'ㅊ': 'ちゃ', 'ㅋ': 'か',
                'ㅌ': 'た', 'ㅍ': 'ぱ'
            },
            "종료": {
                'ㄱ': 'く', 'ㄲ': 'っく', 'ㄷ': 'と', 
                'ㄸ': 'っと', 'ㅂ': 'ぷ', 'ㅃ': 'っぷ', 
                'ㅅ': 'す', 'ㅆ': 'っす', 'ㅈ': 'じ', 
                'ㅉ': 'っじ', 'ㅊ': 'ち', 'ㅋ': 'く',
                'ㅌ': 'と', 'ㅍ': 'ぷ'
            }
        }
        
        # 전이 확률 행렬 (강화된 HMM)
        self.transition_prob = np.array([
            [0.6, 0.35, 0.05],  # 시작 -> (시작, 중간, 끝)
            [0.1, 0.75, 0.15],  # 중간 -> (시작, 중간, 끝)
            [0.25, 0.15, 0.6]   # 끝 -> (시작, 중간, 끝)
        ])
        
        # 한국어 음절 정규식 패턴
        self.ko_syllable_pattern = re.compile(r'[가-힣]')
        self.final_consonant_pattern = re.compile(r'[ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ]$')
        
        # 학습 데이터 (확장된 데이터셋)
        self.training_data = {
            '안녕하세요': 'あんにょんはせよ',
            '감사합니다': 'かむさはむにだ',
            '한국어': 'はんぐごお',
            '좋아요': 'じょあよ',
            '네': 'ね',
            '아니요': 'あによ',
            '사랑해요': 'さらんへよ',
            '미안해요': 'みあんへよ',
            '어서오세요': 'おそおせよ',
            '잘 지내세요': 'ちゃる ちねせよ',
            '잘 가세요': 'ちゃる かせよ',
            '배고파요': 'ぺごぱよ',
            '행복해요': 'へんぼけよ',
            '슬퍼요': 'するぽよ',
            '화났어요': 'ふぁなそよ',
            '질문': 'ちるむん',
            '응답': 'うんだぷ',
            '말씀하세요': 'まるすむはせよ',
            '이해했어요': 'いへへそよ',
            '모르겠어요': 'もるげそよ',
            '도와드릴게요': 'とわどりるげよ',
            '좋은 아침': 'ちょうん あちむ',
            '좋은 하루': 'ちょうん はる',
            '잘 자요': 'ちゃる ちゃよ',
            '반가워요': 'ぱんがおよ',
            '괜찮아요': 'けんちゃなよ',
            # 추가 학습 데이터
            '안녕': 'あんにょん',
            '고마워': 'こまお',
            '미안해': 'みあんへ',
            '사랑해': 'さらんへ',
            '반가워': 'ぱんがお',
            '괜찮아': 'けんちゃな',
            '재미있어': 'ちぇみいそ',
            '그래요': 'くれよ',
            '알겠어요': 'あるげそよ',
            '어디에요': 'おでぃえよ',
            '뭐해요': 'むぇへよ',
            '언제': 'おんじぇ',
            '어떻게': 'おとけ',
            '왜': 'うぇ',
            '누구': 'ぬぐ',
            '무엇': 'むおっ',
            '시작': 'しじゃく',
            '끝': 'くっ',
            '먹어요': 'もごよ',
            '마셔요': 'ましょよ',
            '보여요': 'ぼよよ',
            '들어요': 'とろよ',
            '해요': 'へよ',
            '가요': 'かよ',
            '와요': 'わよ',
            '자요': 'じゃよ',
            '일어나요': 'いろなよ',
            '앉아요': 'あんじゃよ',
            '서요': 'そよ',
            '울어요': 'うろよ',
            '웃어요': 'うそよ'
        }
        
        # 발음 문맥 고려 (문맥에 따른 발음 변화)
        self.context_rules = [
            (r'([가-힣])ㄹ ([ㄱㄷㅂㅅㅈ])', r'\1르 \2'),  # ㄹ 받침 + 자음 시작 단어
            (r'([가-힣])ㄴ ([ㄱㄷㅂㅅㅈ])', r'\1느 \2'),  # ㄴ 받침 + 자음 시작 단어
            (r'([가-힣])ㅁ ([ㄱㄷㅂㅅㅈ])', r'\1므 \2'),  # ㅁ 받침 + 자음 시작 단어
            (r'([가-힣])은 ([ㅇㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ])', r'\1은 \2'),  # 은 + 모음
            (r'([가-힣])는 ([ㅇㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ])', r'\1는 \2'),  # 는 + 모음
            (r'([가-힣])을 ([ㅇㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ])', r'\1을 \2'),  # 을 + 모음
            (r'([가-힣])를 ([ㅇㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ])', r'\1를 \2'),  # 를 + 모음
        ]
        
    def decompose_hangul(self, char):
        """한글 글자를 초성, 중성, 종성으로 분해"""
        if not re.match(r'[가-힣]', char):
            return char  # 한글이 아니면 그대로 반환
        
        code = ord(char) - ord('가')
        
        # 초성, 중성, 종성 분리
        a = code // (21 * 28)  # 초성
        b = (code % (21 * 28)) // 28  # 중성
        c = code % 28  # 종성
        
        return (self.get_cho(a), self.get_jung(b), self.get_jong(c))
    
    def get_cho(self, code):
        """초성 반환"""
        if code < 0 or code >= len(self.get_cho_list()):
            return ''
        return self.get_cho_list()[code]
    
    def get_jung(self, code):
        """중성 반환"""
        if code < 0 or code >= len(self.get_jung_list()):
            return ''
        return self.get_jung_list()[code]
    
    def get_jong(self, code):
        """종성 반환"""
        if code < 0 or code >= len(self.get_jong_list()):
            return ''
        return self.get_jong_list()[code]
    
    def get_cho_list(self):
        """초성 목록"""
        return ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    
    def get_jung_list(self):
        """중성 목록"""
        return ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    
    def get_jong_list(self):
        """종성 목록"""
        return ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    
    def segment_korean(self, text: str) -> List[str]:
        """한국어 텍스트를 의미 단위로 분할 (강화된 방식)"""
        # 단순한 규칙 기반 분할 + 문맥 패턴 적용
        for pattern, replacement in self.context_rules:
            text = re.sub(pattern, replacement, text)
            
        # 문장 패턴 기반 분할
        segments = []
        buffer = ""
        
        for char in text:
            if char in '.,!?;: ':
                if buffer:
                    segments.append(buffer)
                    buffer = ""
                continue
            
            buffer += char
            
            # 단어, 어미, 조사 패턴 탐지
            if len(buffer) >= 3:
                # 단어 경계 확인 (어미/조사 기준)
                for key in sorted(self.training_data.keys(), key=len, reverse=True):
                    if buffer.endswith(key) and len(buffer) > len(key):
                        # 특정 단어나 어미 발견
                        segments.append(buffer[:-len(key)])
                        segments.append(key)
                        buffer = ""
                        break
        
        if buffer:
            segments.append(buffer)
            
        # 지나치게 짧은 세그먼트 병합
        merged_segments = []
        temp = ""
        
        for segment in segments:
            if len(segment) <= 1 and temp:
                temp += segment
            else:
                if temp:
                    merged_segments.append(temp)
                temp = segment
        
        if temp:
            merged_segments.append(temp)
            
        return merged_segments if merged_segments else segments
    
    def advanced_hmm_transform(self, segment: str) -> str:
        """강화된 HMM 기반 변환"""
        if segment in self.training_data:
            return self.training_data[segment]
        
        # 음절 분석 및 변환
        result = ""
        current_state = 0  # 시작 상태
        
        for i, char in enumerate(segment):
            # 음절 분해
            if re.match(r'[가-힣]', char):
                cho, jung, jong = self.decompose_hangul(char)
                
                # 상태에 따른 발음 변화 적용
                if i == 0:
                    state_type = "시작"
                elif i == len(segment) - 1:
                    state_type = "종료"
                else:
                    state_type = "중간"
                
                # 초성 변환 (상태 고려)
                if cho in self.ko_state_patterns[state_type]:
                    jp_cho = self.ko_state_patterns[state_type][cho]
                else:
                    jp_cho = self.ko_to_ja_phoneme_map.get(cho, '')
                
                # 중성 변환
                jp_jung = self.ko_to_ja_phoneme_map.get(jung, '')
                
                # 종성 변환
                if jong:
                    jp_jong = self.ko_to_ja_phoneme_map.get(jong + '_종', '')
                else:
                    jp_jong = ''
                
                # HMM 상태 전이
                next_state_probs = self.transition_prob[current_state]
                next_state = np.random.choice(3, p=next_state_probs)
                current_state = next_state
                
                # 상태에 따른 발음 변화 확률
                if next_state == 0:  # 시작 상태: 자음 강조
                    jp_result = jp_cho + jp_jung + jp_jong
                elif next_state == 1:  # 중간 상태: 일반 발음
                    jp_result = jp_cho + jp_jung + jp_jong
                else:  # 종료 상태: 종성 강조
                    if jp_jong:
                        jp_result = jp_cho + jp_jung + jp_jong
                    else:
                        jp_result = jp_cho + jp_jung
                
                result += jp_result
            else:
                # 한글이 아닌 문자는 그대로 추가
                result += char
        
        return result
    
    def process(self, text: str) -> str:
        """한국어 텍스트를 VoiceVox에 적합한 일본어로 변환 (강화된 버전)"""
        # 전체 문장에서 특수 단어 탐색 및 변환
        for korean, japanese in sorted(self.training_data.items(), key=lambda x: len(x[0]), reverse=True):
            if korean in text:
                text = text.replace(korean, f"<PLACEHOLDER_{hash(korean)}>")
        
        # 세그먼트 분할
        segments = self.segment_korean(text)
        
        # 각 세그먼트 변환
        result_segments = []
        
        for segment in segments:
            # 플레이스홀더 복원
            for korean, japanese in self.training_data.items():
                placeholder = f"<PLACEHOLDER_{hash(korean)}>"
                if placeholder in segment:
                    segment = segment.replace(placeholder, japanese)
                    break
            
            # 일반 세그먼트 변환
            if not any(placeholder in segment for placeholder in [f"<PLACEHOLDER_{hash(k)}>" for k in self.training_data.keys()]):
                transformed = self.advanced_hmm_transform(segment)
                result_segments.append(transformed)
            else:
                result_segments.append(segment)
        
        # 최종 결과 조합
        result = ' '.join(result_segments)
        
        # 남은 플레이스홀더 복원
        for korean, japanese in self.training_data.items():
            placeholder = f"<PLACEHOLDER_{hash(korean)}>"
            if placeholder in result:
                result = result.replace(placeholder, japanese)
        
        return result

# VoiceVox 전처리기 통합
def preprocess_korean_for_voicevox(text: str) -> str:
    """한국어 텍스트를 VoiceVox에 맞게 전처리"""
    processor = KoreanHMMProcessor()
    return processor.process(text) 