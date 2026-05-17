import chromadb
from collections import Counter
from typing import List, Dict
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
import numpy as np

class TagTransitionEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.vocab = ["O", "B-ENT", "I-ENT"]
        self.vocab_size = len(self.vocab)
        self.vector_dim = self.vocab_size + (self.vocab_size ** 2)

        
        
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            raw_tags = text.strip().split()
            
            clean_tags = [
                "B-ENT" if t.startswith("B-") and t[2:] in ["PER", "LOC", "ORG"] else 
                "I-ENT" if t.startswith("I-") and t[2:] in ["PER", "LOC", "ORG"] else 
                "O" 
                for t in raw_tags
            ]
            
            unigram_counts = Counter(clean_tags)
            unigram_vec = [unigram_counts.get(v, 0) for v in self.vocab]
            
            bigram_counts = Counter(zip(clean_tags[:-1], clean_tags[1:]))
            bigram_vec = [bigram_counts.get((v1, v2), 0) for v1 in self.vocab for v2 in self.vocab]
            
            feature_vector = np.array(unigram_vec + bigram_vec, dtype=np.float32)
            norm = np.linalg.norm(feature_vector)
            if norm > 0:
                feature_vector = feature_vector / norm
                
            embeddings.append(feature_vector.tolist())
            
        return embeddings



class RAGVotingEngine:
    def __init__(self, db_path="./chroma_db", use_topology=True):
        self.embedding_function = TagTransitionEmbeddingFunction() if use_topology else None
        
        self.client = chromadb.PersistentClient(path=db_path)
        
        try:
            self.client.delete_collection(name="ner_topology_memory")
        except:
            pass
            
        self.collection = self.client.create_collection(
            name="ner_topology_memory",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
    
        
    def build_knowledge_base(self, hard_examples: List[Dict]):
       
        documents = [ex["dirty"] for ex in hard_examples]
        metadatas = [{"clean_tags": ex["clean"]} for ex in hard_examples]
        ids = [ex["id"] for ex in hard_examples]
        
        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def retrieve_hard_examples(self, dirty_tags: List[str], top_k: int = 3) -> List[str]:
        query_text = " ".join(dirty_tags) 
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        
        if not results['metadatas'][0]:
            return []
            
        retrieved_cleans = [meta["clean_tags"] for meta in results['metadatas'][0]]
        return retrieved_cleans

    def majority_voting_with_rag(self, 
                                 candidate_paths: List[List[str]], 
                                 rag_weights: List[float],
                                 entity_boost: float = 1.35) -> List[str]: 

        if len(candidate_paths) != len(rag_weights):
            raise ValueError("路径数量必须与权重数量一致！")

        sequence_length = len(candidate_paths[0])
        final_voted_tags = []

        VALID_ENTITIES = {"PER", "LOC", "ORG"}

        for t in range(sequence_length):
            vote_tally = Counter()
            
            for i, path in enumerate(candidate_paths):
                tag = path[t]
                weight = rag_weights[i]
                
                is_valid_entity = False
                if tag != "O" and "-" in tag:
                    prefix, ent_type = tag.split("-", 1)
                    if prefix in {"B", "I"} and ent_type in VALID_ENTITIES:
                        is_valid_entity = True
                
                if is_valid_entity:
                    vote_tally[tag] += (weight * entity_boost) 
                else:
                    vote_tally["O"] += weight 
            
            if not vote_tally:
                winning_tag = "O"
            else:
                winning_tag = vote_tally.most_common(1)[0][0]
                
            final_voted_tags.append(winning_tag)

        return final_voted_tags



if __name__ == "__main__":
    engine = RAGVotingEngine()
    
    paths = [
        ["O", "I-PER", "I-PER"], 
        ["B-PER", "I-PER", "O"], 
        ["B-PER", "I-PER", "I-PER"], 
        ["O", "O", "O"],         
        ["B-PER", "I-PER", "O"]  
    ]
    
    weights = [0.1, 0.9, 0.4, 0.0, 0.8] 
    
   
    final_output = engine.majority_voting_with_rag(paths, weights)
    print(f" 最终 RAG 加权投票结果: {final_output}") 
