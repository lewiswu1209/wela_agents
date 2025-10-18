
from typing import Dict
from typing import List
from typing import TypeVar
from typing import Callable
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Record
from qdrant_client.models import Distance
from qdrant_client.models import PointStruct
from qdrant_client.models import VectorParams
from qdrant_client.models import ExtendedPointId
from qdrant_client.conversions.common_types import ScoredPoint

from wela_agents.memory.memory import Memory
from wela_agents.reranker.reranker import Reranker
from wela_agents.embedding.embedding import Embedding

T = TypeVar("T")

class QdrantMemory(Memory[T]):

    def __init__(self,
        memory_key: str,
        embedding: Embedding,
        reranker: Reranker,
        qdrant_client: QdrantClient,
        get_text: Callable[[T], str],
        vector_size = 512,
        score_threshold: Optional[float] = None,
        limit: int=10,
        window_size = 10
    ) -> None:
        super().__init__(memory_key)
        self.__embedding = embedding
        self.__reranker = reranker
        self.__client: QdrantClient = qdrant_client
        self.__get_text = get_text
        self.__vector_size = vector_size
        self.__score_threshold: Optional[float] = score_threshold
        self.__limit: int = limit
        self.__window_size = window_size

        if not self.__client.collection_exists(collection_name=self.memory_key):
            self.__client.create_collection(
                collection_name=self.memory_key,
                vectors_config=VectorParams(size=self.__vector_size, distance=Distance.COSINE)
            )

    def save_content(self, content: T) -> None:
        text = self.__get_text(content)
        if text:
            for embedding in self.__embedding.embed([text]):
                saved_memory_count = self.__client.count(collection_name=self.memory_key).count
                self.__client.upsert(
                    collection_name=self.memory_key,
                    points = [PointStruct(
                        id = saved_memory_count,
                        vector = [float(x) for x in embedding],
                        payload = content
                    )]
                )

    def get_contents(self, contents: List[T]) -> List[T]:
        results: List[ScoredPoint] = []
        for content in contents:
            text = self.__get_text(content)
            if text:
                for query_vector in self.__embedding.embed([text]):
                    search_results = self.__client.search(
                        collection_name=self.memory_key,
                        query_vector=query_vector,
                        limit=self.__limit,
                        score_threshold = self.__score_threshold,
                    )
                    documents = [self.__get_text(search_result.payload) for search_result in search_results]
                    for rerank_result in self.__reranker.rerank(text, documents):
                        item: ScoredPoint = search_results[rerank_result["index"]]
                        item.score = rerank_result["score"]
                        results.append(item)

        saved_memory_count = self.__client.count(collection_name=self.memory_key).count
        latest_ids = list(range(0, saved_memory_count))[-self.__window_size:]
        records: List[Record] = self.__client.retrieve(
            collection_name=self.memory_key,
            ids=latest_ids
        )
        for record in records:
            results.append(
                ScoredPoint(id=record.id, version=1, score=1.0, payload=record.payload, vector=record.vector)
            )

        deduplicated: Dict[ExtendedPointId, ScoredPoint] = {}
        for result in results:
            if result.id not in deduplicated or result.score > deduplicated[result.id].score:
                deduplicated[result.id] = result
        results = sorted(deduplicated.values(), key=lambda x: x.score, reverse=True)[:self.__limit]
        results = sorted(results, key=lambda x: x.id, reverse=False)
        return [i.payload for i in results]

    def reset_memory(self) -> None:
        self.__client.delete_collection(self.memory_key)

        self.__client.create_collection(
            collection_name=self.memory_key,
            vectors_config=VectorParams(size=self.__vector_size, distance=Distance.COSINE),
        )

__all__ = [
    "QdrantMemory"
]
