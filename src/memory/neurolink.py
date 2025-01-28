import os
from typing import (
    Any,
    List,
    Dict,
    Union,
    Tuple,
)
from collections import deque

from src.registry import MEMORY
from src.memory.base import VectorStore, Image
from src.memory.faiss_store import FaissVectorStore
from src.memory.basic_memory import MemoryUnit

@MEMORY.register_module(force=True)
class MemoryInterface:
    '''Interface to interact with each memory storage unit.'''
    
    def __init__(
        self,
        root,
        symbols: List[str],
        memory_path: str,
        embedding_dim: int,
        max_recent_steps = 5,
        workdir = None,
        tag = None,
    ) -> None:
        """
        Initialize a MemoryInterface instance.

        This class manages multiple types of memory (market intelligence,
        low-level reflection, high-level reflection) for a list of symbols.
        Each memory type and symbol combination has its own BasicMemory and vector store.

        Args:
            root: The root directory where memory files are stored.
            symbols: A list of asset symbols (e.g., ['BTC', 'ETH']).
            memory_path: Relative path (under root/workdir/tag) where memory files are stored.
            embedding_dim: The dimension of the embedding vectors.
            max_recent_steps: The maximum number of recent history items to keep in memory.
            workdir: Optional subdirectory under root.
            tag: Optional identifier appended to the directory path.
        """
        self.root = root
        self.symbols = symbols
        self.embedding_dim = embedding_dim
        self.max_recent_steps = max_recent_steps
        self.workdir = workdir
        self.tag = tag
        
        # Create the relative memory path
        self.memory_path = os.path.join(self.root, self.workdir, memory_path)
        os.makedirs(self.memory_path, exist_ok=True)
        
        # Initialize memory stores that will contain memories for each symbol
        self.market_intelligence_memories = {}
        self.low_level_reflection_memories = {}
        self.high_level_reflection_memories = {}
        self._init_memories()
        
        # Initializes recent memory stores for each memory type
        self.market_intelligence_recent_memories = {}
        self.low_level_reflection_recent_memories = {}
        self.high_level_reflection_recent_memories = {}
        self._init_recent_memories()
        
    def _init_memories(self) -> None:
        '''
        Initialize MemoryUnit instances for each symbol and memory type.

        For each symbol, creates directories and initializes a FAISS vector store
        and a MemoryUnit object for market intelligence, low-level reflection,
        and high-level reflection.
        '''
        for symbol in self.symbols:
            
            # Market Intelligence memory setup
            if symbol not in self.market_intelligence_memories:
                mi_path = os.path.join(self.memory_path, symbol, "market_intelligence")
                os.makedirs(mi_path, exist_ok=True)
                vecstore = FaissVectorStore(embedding_dim=self.embedding_dim,
                                            memory_path=mi_path)
                self.market_intelligence_memories[symbol] = MemoryUnit(memory_path=mi_path,
                                                                       vectorstore=vecstore)
                
            # Low-level reflection memory setup
            if symbol not in self.low_level_reflection_memories:
                llr_path = os.path.join(self.memory_path, symbol, "low_level_reflection")
                os.makedirs(llr_path, exist_ok=True)
                vecstore = FaissVectorStore(embedding_dim=self.embedding_dim, 
                                            memory_path=llr_path)
                self.low_level_reflection_memories[symbol] = MemoryUnit(memory_path=llr_path,
                                                                        vectorstore=vecstore)
                
            # High-level reflection memory setup
            if symbol not in self.high_level_reflection_memories:
                hlr_path = os.path.join(self.memory_path, symbol, "high_level_reflection")
                os.makedirs(hlr_path, exist_ok=True)
                vecstore = FaissVectorStore(embedding_dim=self.embedding_dim,
                                            memory_path=hlr_path)
                self.high_level_reflection_memories[symbol] = MemoryUnit(memory_path=hlr_path,
                                                                         vectorstore=vecstore)
                
    def _init_recent_memories(self) -> None:
        """
        Initialize recent memories for each symbol and memory type.
        """
        for symbol in self.symbols:
            
            # Market Intelligence recent memories
            if symbol not in self.market_intelligence_recent_memories:
                self.market_intelligence_recent_memories[symbol] = deque(maxlen=self.max_recent_steps)
            
            # Low-level Reflection recent memories
            if symbol not in self.low_level_reflection_recent_memories:
                self.low_level_reflection_recent_memories[symbol] = deque(maxlen=self.max_recent_steps)
            
            # High-level Reflection recent memories
            if symbol not in self.high_level_reflection_recent_memories:
                self.high_level_reflection_recent_memories[symbol] = deque(maxlen=self.max_recent_steps)
            
    def get_memory(self, memory_type: str, symbol: str) -> MemoryUnit:
        """
        Get a MemoryUnit object for a given memory_type and symbol.

        Args:
            memory_type: The type of memory. One of ["market_intelligence", "low_level_reflection", "high_level_reflection"].
            symbol: The symbol for which to retrieve the memory.

        Returns:
            The corresponding MemoryUnit instance.

        Raises:
            AssertionError: If the type is invalid.
        """
        return self._get_memory(memory_type, symbol)

    def _get_memory(self, memory_type: str, symbol: str) -> MemoryUnit:
        """
        Internal method to retrieve the MemoryUnit for a given memory_type and symbol.

        Args:
            memory_type: Memory type, must be one of ["market_intelligence", "low_level_reflection", "high_level_reflection"].
            symbol: The symbol for which the memory is retrieved.

        Returns:
            The MemoryUnit instance associated with the given type and symbol.
        """
        assert memory_type in ["market_intelligence", "low_level_reflection", "high_level_reflection"],\
            f"memory_type = {memory_type} should be one of ['market_intelligence', 'low_level_reflection', 'high_level_reflection']."

        if memory_type == "market_intelligence":
            return self.market_intelligence_memories[symbol]
        elif memory_type == "low_level_reflection":
            return self.low_level_reflection_memories[symbol]
        elif memory_type == "high_level_reflection":
            return self.high_level_reflection_memories[symbol]
        
    def _get_recent_history(self, memory_type: str, symbol: str) -> deque:
        """
        Internal method to retrieve the recent history deque for a given memory_type and symbol.

        Args:
            memory_type: Memory type, one of ["market_intelligence", "low_level_reflection", "high_level_reflection"].
            symbol: The symbol for which the recent history is retrieved.

        Returns:
            A deque containing recent history entries for the specified type and symbol.
        """
        assert memory_type in ["market_intelligence", "low_level_reflection", "high_level_reflection"], \
            f"memory_type = {memory_type} should be one of ['market_intelligence', 'low_level_reflection', 'high_level_reflection']."

        if memory_type == "market_intelligence":
            return self.market_intelligence_recent_memories[symbol]
        elif memory_type == "low_level_reflection":
            return self.low_level_reflection_recent_memories[symbol]
        elif memory_type == "high_level_reflection":
            return self.high_level_reflection_recent_memories[symbol]
        
    def add_memory(
        self,
        memory_type: str,
        symbol: str,
        data: Dict,
        embedding_key: str,
    ) -> None:
        """
        Add a new memory entry for a specific memory_type and symbol.

        This adds the provided data (which should contain an embedding under 'embedding_key')
        to the BasicMemory. The data and its embedding will be stored for future similarity queries.

        Args:
            memory_type: The memory type ("market_intelligence", "low_level_reflection", or "high_level_reflection").
            symbol: The symbol for which memory is stored.
            data: A dictionary containing the data to store. Must include embedding_key.
            embedding_key: The key in data that corresponds to the embedding vector.
        """
        memory = self._get_memory(memory_type, symbol)
        memory.add(data=data, embedding_key=embedding_key)
        print(f"Add memory for {memory_type} {symbol}.")
        
    def query_memory(
        self,
        memory_type: str,
        symbol: str,
        data: Dict,
        embedding_query: str,
        top_k: int = 3
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Query the memory for similar items.

        Given a query embedding (found at embedding_query in data), this method returns
        the top_k most similar items that have been stored in the memory for the specified
        memory_type and symbol.

        Args:
            memory_type: The memory type ("market_intelligence", "low_level_reflection", or "high_level_reflection").
            symbol: The symbol to query against.
            data: A dictionary containing the query embedding under embedding_query.
            embedding_query: The key in data that holds the query embedding.
            top_k: The number of top similar items to return.

        Returns:
            A tuple containing:
            - A list of dictionaries representing the retrieved items.
            - A list of floats representing their similarity scores.
        """
        memory = self._get_memory(memory_type, symbol)
        res = memory.query(
            data=data,
            embedding_query=embedding_query,
            top_k=top_k,
        )
        print(f"Query memory for {memory_type} {symbol}.")
        return res
    
    def add_recent_history(
        self,
        memory_type: str,
        symbol: str,
        data: Dict,
    ) -> None:
        """
        Add a piece of recent history data for a given memory_type and symbol.

        This data won't go through the vector store. It's just kept in a short-term
        buffer for quick and simple retrieval.

        Args:
            memory_type: The memory type ("market_intelligence", "low_level_reflection", or "high_level_reflection").
            symbol: The symbol for which to add recent history.
            data: A dictionary containing the history item.
        """
        recent_history = self._get_recent_history(memory_type, symbol)
        recent_history.append(data)
        print(f"Add recent history for {memory_type} {symbol}.")
        
    def get_recent_history(
        self,
        memory_type: str,
        symbol: str,
        k: int = 1,
    ) -> List[Any]:
        """
        Retrieve up to k most recent history items for the given memory_type and symbol.

        Args:
            memory_type: The memory type ("market_intelligence", "low_level_reflection", or "high_level_reflection").
            symbol: The symbol to retrieve recent history for.
            k: How many recent items to retrieve, must not exceed max_recent_steps.

        Returns:
            A list of the most recent k items of that type and symbol.
        """
        assert k <= self.max_recent_steps, f"k = {k} should be less than or equal to max_recent_steps = {self.max_recent_steps}."

        # Grab recent history for symbol
        recent_history_ = self._get_recent_history(memory_type, symbol)
        recent_history = list(recent_history_)
        if len(recent_history) < k:
            res = recent_history
        else:
            res = recent_history[-k:]

        print(f"Get recent history for {memory_type} {symbol}.")
        return res
    
    def load_local(
        self,
        memory_path: str = None,
    ) -> None:
        """
        Load all memories from the local file system.

        This loads both the vector stores and the memory dictionaries for each symbol and memory type.
        If loading fails for a particular type, it prints an error message and continues.

        Args:
            memory_path: Optional override of the memory path. If not provided, uses self.memory_path.
        """
        if memory_path is None:
            memory_path = self.memory_path

        # Load market intelligence, low-level reflection, and high-level reflection memories
        for symbol in self.symbols:
            
            # Market intelligence loading
            try:
                path = os.path.join(memory_path, symbol, "market_intelligence")
                print(path)
                os.makedirs(path, exist_ok=True)
                
                # Load Vector Store
                vecstore = FaissVectorStore(memory_path=path, embedding_dim=self.embedding_dim)
                vecstore.load_local(memory_path=path, embedding_dim=self.embedding_dim)
                print(f"symbols: {symbol}, memory_path: {path}, vecstore length: {vecstore.index.ntotal}")
                
                # Load Memories
                self.market_intelligence_memories[symbol].load_local(
                    memory_path=path,
                    vectorstore=vecstore,
                )
            except Exception as e:
                print(f"Failed to load market_intelligence_memories: {e}")

            # Low-level reflection
            try:
                path = os.path.join(memory_path, symbol, "low_level_reflection")
                os.makedirs(path, exist_ok=True)
                
                # Load Vector Store
                vecstore = FaissVectorStore(memory_path=path, embedding_dim=self.embedding_dim)
                vecstore.load_local(memory_path=path, embedding_dim=self.embedding_dim)
                print(f"symbols: {symbol}, memory_path: {path}, vecstore length: {vecstore.index.ntotal}")
                
                # Load Memories
                self.low_level_reflection_memories[symbol].load_local(
                    memory_path=path,
                    vectorstore=vecstore,
                )
            except Exception as e:
                print(f"Failed to load low_level_reflection_memorys: {e}")

            # High-level reflection
            try:
                path = os.path.join(memory_path, symbol, "high_level_reflection")
                os.makedirs(path, exist_ok=True)
                
                # Load Vector Store
                vecstore = FaissVectorStore(memory_path=path, embedding_dim=self.embedding_dim)
                vecstore.load_local(memory_path=path, embedding_dim=self.embedding_dim)
                print(f"symbols: {symbol}, memory_path: {path}, vecstore length: {vecstore.index.ntotal}")
                
                # Load Memories
                self.high_level_reflection_memories[symbol].load_local(
                    memory_path=path,
                    vectorstore=vecstore,
                )
            except Exception as e:
                print(f"Failed to load high_level_reflection_memorys: {e}")
                
    def save_local(self, memory_path: str = None) -> None:
        """
        Save all memories and vector stores to the local file system.

        Args:
            memory_path: Optional override of the memory path. If not provided, uses self.memory_path.
        """
        if memory_path is None:
            memory_path = self.memory_path

        for symbol in self.symbols:
            # Save market intelligence memory
            path = os.path.join(memory_path, symbol, "market_intelligence")
            os.makedirs(path, exist_ok=True)
            self.market_intelligence_memories[symbol].save_local(path)

            # Save low-level reflection memory
            path = os.path.join(memory_path, symbol, "low_level_reflection")
            os.makedirs(path, exist_ok=True)
            self.low_level_reflection_memories[symbol].save_local(path)

            # Save high-level reflection memory
            path = os.path.join(memory_path, symbol, "high_level_reflection")
            os.makedirs(path, exist_ok=True)
            self.high_level_reflection_memories[symbol].save_local(path)