"""Description: This file contains the implementation of the LLM class using llama.cpp.
This class provides a stateless interface to llama.cpp for language generation.
"""

import asyncio
from typing import AsyncIterator, List, Dict, Any
from llama_cpp import Llama
from loguru import logger

from .stateless_llm_interface import StatelessLLMInterface


class LLM(StatelessLLMInterface):
    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,  # -1: 자동 감지, 0: CPU만, 양수: GPU 레이어 수
        main_gpu: int = 0,  # 사용할 주 GPU 인덱스
        n_ctx: int = 4096,  # 컨텍스트 윈도우 크기 (최대 길이)
        n_batch: int = 512,  # 배치 크기 증가 (기본값 8 -> 512)
        cache_capacity: int = 2000,  # KV 캐시 크기 증가
        f16_kv: bool = True,  # KV 캐시를 FP16으로 저장
        use_mlock: bool = True,  # mlock 사용 (시스템 메모리에 모델 잠금)
        mmap: bool = True,  # 메모리 매핑 사용
        use_mmap: bool = True,  # 메모리 매핑 사용 (최신 버전용)
        seed: int = 42,  # 예측 가능한 결과를 위한 시드
        verbose: bool = False,  # 디버깅 정보 출력
        **kwargs,
    ):
        """
        Initializes a stateless instance of the LLM class using llama.cpp.

        Parameters:
        - model_path (str): Path to the GGUF model file
        - n_gpu_layers (int): Number of layers to offload to GPU (-1 for auto)
        - main_gpu (int): Index of GPU to use for main computation
        - n_ctx (int): Context window size
        - n_batch (int): Batch size for prompt evaluation (higher = faster but uses more VRAM)
        - cache_capacity (int): KV cache size (higher = less recomputation but uses more memory)
        - f16_kv (bool): Use FP16 for KV cache (faster and less memory)
        - use_mlock (bool): Lock model in memory (prevents swapping)
        - mmap (bool): Use memory mapping for model loading
        - use_mmap (bool): Use memory mapping (newer parameter)
        - seed (int): Random seed for reproducibility
        - verbose (bool): Print debug information
        - **kwargs: Additional arguments passed to Llama constructor
        """
        logger.info(f"Initializing llama cpp with model path: {model_path}")
        self.model_path = model_path
        # 최적화 옵션
        optimized_args = {
            "n_gpu_layers": n_gpu_layers,
            "main_gpu": main_gpu,
            "n_ctx": n_ctx,
            "n_batch": n_batch, 
            "cache_capacity": cache_capacity,
            "f16_kv": f16_kv,
            "use_mlock": use_mlock,
            "mmap": mmap,
            "use_mmap": use_mmap,
            "seed": seed,
            "verbose": verbose
        }
        
        # kwargs에 포함된 값이 있으면 덮어쓰지 않음
        for key, val in optimized_args.items():
            if key not in kwargs:
                kwargs[key] = val
        
        logger.info(f"LLM optimized arguments: {kwargs}")
        
        try:
            self.llm = Llama(model_path=model_path, **kwargs)
        except Exception as e:
            logger.critical(f"Failed to initialize Llama model: {e}")
            raise

    async def chat_completion(
        self, messages: List[Dict[str, Any]], system: str = None
    ) -> AsyncIterator[str]:
        """
        Generates a chat completion using llama.cpp asynchronously.

        Parameters:
        - messages (List[Dict[str, Any]]): The list of messages to send to the model.
        - system (str, optional): System prompt to use for this completion.

        Yields:
        - str: The content of each chunk from the model response.
        """
        logger.debug(f"Generating completion for messages: {messages}")

        try:
            # Add system prompt if provided
            messages_with_system = messages
            if system:
                messages_with_system = [
                    {"role": "system", "content": system},
                    *messages,
                ]

            # 응답 생성 최적화 파라미터 설정
            completion_params = {
                "messages": messages_with_system,
                "stream": True,
                "temperature": 0.7,  # 낮은 온도 = 더 빠른 응답
                "top_k": 40,          # top_k 샘플링 제한
                "top_p": 0.95,        # top_p 샘플링 제한
                "repeat_penalty": 1.1  # 반복 패널티 설정
            }
            
            # Create chat completion in a separate thread to avoid blocking
            chat_completion = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.create_chat_completion(**completion_params),
            )

            # Process chunks
            for chunk in chat_completion:
                if chunk.get("choices") and chunk["choices"][0].get("delta"):
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        yield content

        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise
