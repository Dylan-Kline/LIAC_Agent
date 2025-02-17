from src.memory import MemoryInterface
from src.provider import EmbeddingProvider
from src.query import QUERY_TYPES
from typing import Dict, Any, List

class DiverseQuery():
    def __init__(self,
                 memory: MemoryInterface,
                 provider: EmbeddingProvider,
                 top_k: int = 5):
        self.memory = memory
        self.provider = provider
        self.top_k = top_k
        
    def query(self,
              params: Dict = None,
              query_types: List[str] = ["plain", "short_term", "medium_term", "long_term"],
              top_k: int = None):
        return self.diverse_query(params, query_types=query_types, top_k=top_k)
    
    def diverse_query(self,
                      params: Dict,
                      query_types: List[str] = ["plain", "short_term", "medium_term", "long_term"],
                      top_k: int = None):
        top_k = top_k if top_k is not None else self.top_k
        
        type = params["type"]
        symbol = params["symbol"]
        
        result = {}
        for query_type in query_types:
            query_text = str(QUERY_TYPES[query_type](params))
            embedding = self.provider.embed_query(query_text)
            query_items, _ = self.memory.query_memory(memory_type=type,
                                                      symbol=symbol,
                                                      data={"embedding": embedding},
                                                      embedding_query="embedding",
                                                      top_k=top_k)
            
            if len(query_items) == 0:
                query_items = []
            
            result[query_type] = {
                "query_text": query_text,
                "query_items": query_items
            }
            
        return result