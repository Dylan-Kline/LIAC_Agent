from typing import (
    List,
    Dict,
    Union,
    Optional,
    Tuple,
    Any,
)

import time
import json
import os

from src.memory.base import VectorStore, BaseMemory, Image

class MemoryUnit(BaseMemory):
    '''Memory unit that stores metadata.'''
    
    def __init__(
        self,
        memory_path: str,
        vectorstore: VectorStore,
        memory: Optional[Dict] = None,
    ) -> None:
        '''
        Initializes the memory unit.
        
        Args:
            memory_path (str): A path to where you can save and load the memory from your local file system.
            vectorstore (VectorStore): An object that handles storing and retrieving embeddings.
            memory (Optional[Dict]): An optional dictionary. If provided, it initializes the memory with existing data.
            Otherside, it starts empty.
        '''
        if memory is None:
            self.memory = {}
        else:
            self.memory = memory
        
        self.memory_path = memory_path
        self.vectorstore = vectorstore
        
    def add(
        self,
        data: Dict,
        embedding_key: str,
        **kwards,
    ) -> None:
        '''
        Adds the data to the memory.
        
        Args:
            data (Dict): A dictionary containing the information you want to store.
            embedding_key (str): A string that tells the function which part of 'data' contains the embedding.
        '''
        # Create a unique id for the data and store it in memory
        name = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) # unique id of the added data
        self.memory[name] = data
        
        assert embedding_key in data, f"embedding_key {embedding_key} not in data."
        embeddings = data[embedding_key]
        
        self.vectorstore.add_embeddings([name], [embeddings])
        
    def similarity_search(
        self, 
        data: Dict,
        embedding_query: str,
        top_k: int = 3,
        **kwargs,
        ) -> Tuple[List[Dict[str, Any]], List[float]]:
        '''
        Retrieves keys from the vectorstore based on similarity and returns their associated data stored in memory.
        
        Args:
            data (Dict): A dictionary containing the embedding query you want to search with.
            embedding_query_key (str): The key to access the embedding query in 'data'.
        
        Returns:
            Tuple of the following:
            items (List[Dict[str, Any]]): The original data items that were stored in memory.
            scores (List[float]): The associated similarity scores of each item.
        '''
        
        assert embedding_query in data, f"embedding_query {embedding_query} not in data."
        
        query_embedding = data[embedding_query]
        
        try:
            key_and_score = self.vectorstore.similarity_search(query_embedding, top_k)
            items = [self.memory[key] for key, score in key_and_score]
            scores = [score for key, score in key_and_score]
        except:
            items = []
            scores = []
            
        return items, scores
    
    def query(
        self,
        data: Dict,
        embedding_query: str,
        top_k: int = 3,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        '''
        Queries the stored memory using similarity search.
        
        Args:
            data (Dict): A dictionary containing the embedding query you want to search with.
            embedding_query_key (str): The key to access the embedding query in 'data'.
        
        Returns:
            Tuple of the following:
            items (List[Dict[str, Any]]): The original data items that were stored in memory.
            scores (List[float]): The associated similarity scores of each item.
        '''
        items, scores = self.similarity_search(data=data, 
                                               embedding_query=embedding_query,
                                               top_k=top_k,
                                               **kwargs)
        return items, scores
    
    def load_local(
            self,
            memory_path: str = None,
            vectorstore: VectorStore = None,
    ) -> None:
        """Load the memory from the local file."""
        if memory_path is None:
            memory_path = self.memory_path

        with open(os.path.join(memory_path, "memory.json"), "r") as rf:
            memory = json.load(rf)
    
        self.memory_path = memory_path
        self.vectorstore = vectorstore
        self.memory = memory
        
    def save_local(self, memory_path = None) -> None:
        """Save the memory to the local file."""
        if memory_path is None:
            memory_path = self.memory_path

        with open(os.path.join(memory_path, "memory.json"), "w") as f:
            json.dump(self.memory, f, indent=2)
        self.vectorstore.save_local(memory_path)
    