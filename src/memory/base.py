import abc
from typing import (
    Any,
    Iterable,
    List,
    Dict,
    Union,
    Tuple,
    Optional,
)

Image = Any

class VectorStore(abc.ABC):
    '''Interface for vector store'''

    @abc.abstractmethod
    def add_embeddings(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Add embeddings to the vector store.
        
        Args:
            keys: list of metadatas associated with the embedding.
            embeddings: Iterable of embeddings to add to the vector store.
            kwargs: vectorstore specific parameters.
        """
        pass

    @abc.abstractmethod
    def delete(self,
               *args: Any,
               **kwargs: Any,) -> bool:
        """Delete embeddings by their associated keys.
        
        Args:
            keys: List of keys to delete.
            **kwargs: Other keyword arguments if needed.
            
        Returns:
            bool: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        pass
    
    @abc.abstractmethod
    def similarity_search(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> List[Tuple[str, float]]:
        """Grab keys most similar to query.
        
        Args:
            query_embedding (List[float]): The query embedding to find similar embeddings for.
            top_k (int): The number of similar embeddings to extract.
        """
        pass

    @abc.abstractmethod
    def save_local(self,
                    memory_path: str) -> None:
        '''Save index and metadata to disk.
        
        Args:
            memory_path: The path to store the index and metadata.'''
        pass

    @abc.abstractmethod
    def load_local(self,
                   memory_path: str) -> None:
        '''Load index and metadata from disk.
        
        Args:
            memory_path: The path of the vector store index and metadata.'''
        pass

class BaseMemory:
    '''Base class for all memories.'''

    @abc.abstractmethod
    def add(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Add a batch of memories to the vector store.
        
        Args:
            **kwargs: Other keyword arguments.
        """
        pass

    @abc.abstractmethod
    def similarity_search(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        '''
        Perform a similarity search on the vector store to find the most similar entries.
        '''
        pass

    @abc.abstractmethod
    def query_memory(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> List[Union[str, Image]]:
        """
        Query the vector store for similar entries.
        
        Args:
            query: The query to find similar entries for.
            memory_type: The type of memory to search for.
        """
        pass