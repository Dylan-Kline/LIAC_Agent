import math
import os
import backoff
from typing import Dict, List, Any
from copy import deepcopy

from src.prompt import YamlPrompt
from src.asset import ASSET
from src.memory import MemoryInterface
from src.provider import EmbeddingProvider
from src.query import DiverseQuery

class LatestMarketIntelligenceSummaryPrompt(YamlPrompt):
    def __init__(self, template_path: str) -> None:
        super().__init__(template_path)