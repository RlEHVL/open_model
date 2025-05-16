import atexit
import requests
from loguru import logger
from .openai_compatible_llm import AsyncLLM
from typing import List, Dict, Any, AsyncIterator


class OllamaLLM(AsyncLLM):
    def __init__(
        self,
        model: str,
        base_url: str,
        llm_api_key: str = "z",
        organization_id: str = "z",
        project_id: str = "z",
        temperature: float = 0.7,
        keep_alive: float = -1,
        unload_at_exit: bool = True,
        num_gpu: int = 1,
        num_thread: int = 8,
        max_tokens: int = 1024,
        top_k: int = 40,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1,
        mirostat: int = 0,
        use_cache: bool = True,
    ):
        self.keep_alive = keep_alive
        self.unload_at_exit = unload_at_exit
        self.cleaned = False
        self.num_gpu = num_gpu
        self.num_thread = num_thread
        self.repeat_penalty = repeat_penalty
        self.mirostat = mirostat
        self.use_cache = use_cache
        super().__init__(
            model=model,
            base_url=base_url,
            llm_api_key=llm_api_key,
            organization_id=organization_id,
            project_id=project_id,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
        )
        try:
            logger.info("Preloading model for Ollama with optimized settings")
            preload_params = {
                "model": model,
                "keep_alive": keep_alive,
                "options": {
                    "num_gpu": num_gpu,
                    "num_thread": num_thread,
                    "mirostat": mirostat,
                    "use_cache": use_cache,
                }
            }
            preload_url = base_url.replace("/v1", "") + "/api/chat"
            logger.debug(f"Preloading model with params: {preload_params}")
            response = requests.post(preload_url, json=preload_params)
            
            if response.status_code == 200:
                logger.info(f"Ollama model preloaded successfully: {model}")
            else:
                logger.warning(f"Ollama model preload response: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to preload model: {e}")
            logger.critical(
                "Fail to connect to Ollama backend. Is Ollama server running? Try running `ollama list` to start the server and try again.\nThe AI will repeat 'Error connecting chat endpoint' until the server is running."
            )
        except Exception as e:
            logger.error(f"Failed to preload model: {e}")
        if unload_at_exit:
            atexit.register(self.cleanup)

    def __del__(self):
        """Destructor to unload the model"""
        self.cleanup()

    def cleanup(self):
        """Clean up function to unload the model when exitting"""
        if not self.cleaned and self.unload_at_exit:
            logger.info(f"Ollama: Unloading model: {self.model}")
            logger.debug(
                requests.post(
                    self.base_url.replace("/v1", "") + "/api/chat",
                    json={
                        "model": self.model,
                        "keep_alive": 0,
                    },
                )
            )
            self.cleaned = True

    async def chat_completion(
        self, messages: List[Dict[str, Any]], system: str = None
    ) -> AsyncIterator[str]:
        """
        Ollama 모델에 최적화된 매개변수로 응답 생성
        """
        completion_params = {
            "repeat_penalty": self.repeat_penalty,
            "mirostat": self.mirostat,
            "num_gpu": self.num_gpu,
            "num_thread": self.num_thread,
        }
        
        self.response_format = {"options": completion_params}
        
        async for content in await super().chat_completion(messages, system):
            yield content
