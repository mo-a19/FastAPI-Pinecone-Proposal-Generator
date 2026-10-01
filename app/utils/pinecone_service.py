import logging
from typing import List, Dict, Any, Tuple, Optional
from pinecone import Pinecone, ServerlessSpec
from tenacity import retry, wait_exponential, stop_after_attempt
import json

logger = logging.getLogger(__name__)

class PineconeService:
    """Service for managing Pinecone vector database with smart retrieval strategy"""
    
    def __init__(
        self,
        api_key: str,
        environment: str = "us-east-1",
        index_name: str = "proposal-engine",
        namespace: str = "portfolio",
        dimension: int = 3072
    ):
        """
        Initialize Pinecone service
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment region
            index_name: Name of the index
            namespace: Namespace for organizing vectors
            dimension: Dimension of embeddings (3072 for text-embedding-3-large)
        """
        self. pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.namespace = namespace
        self.environment = environment
        self.dimension = dimension
        
        # Initialize or get index
        self.index = self._init_index()
        logger.info(f"Pinecone service initialized - Index: {index_name}, Namespace: {namespace}, Dimension: {dimension}")
    
    def _init_index(self):
        """Initialize Pinecone index if it doesn't exist"""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]
            
            if self.index_name in index_names:
                # Index exists - use it without deleting data
                idx = self.pc.Index(self.index_name)
                idx_info = self.pc.describe_index(self.index_name)
                existing_dim = idx_info.dimension if hasattr(idx_info, 'dimension') else None
                
                if existing_dim and existing_dim != self.dimension:
                    logger.warning(f"Index {self.index_name} has dimension {existing_dim}, but expecting {self.dimension}. Using existing index without modification (preserving data).")
                    # Use existing index without deleting - do not recreate
                    logger.info(f"Using existing index: {self.index_name} to preserve historical data")
                else:
                    logger.info(f"Index {self.index_name} already exists with correct dimension")
                
                return idx
            
            logger.info(f"Creating new index: {self.index_name} with dimension {self.dimension}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",  # Cosine similarity for semantic search
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.environment
                )
            )
            logger.info(f"Index {self.index_name} created successfully with dimension {self.dimension}")
            
            return self.pc.Index(self.index_name)
        except Exception as e:
            logger.error(f"Error initializing index: {str(e)}")
            raise
    
    # ===================== UPSERT OPERATIONS =====================
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def upsert_vectors(
        self,
        vectors: List[Tuple[str, List[float], Dict[str, Any]]],
        batch_size: int = 100
    ) -> int:
        """
        Upsert vectors into Pinecone with smart metadata for retrieval
        
        Format of vectors: (vector_id, embedding, metadata)
        
        Args:
            vectors: List of (id, embedding, metadata) tuples
            batch_size: Batch size for upserting
            
        Returns:
            Number of vectors upserted
        """
        try:
            upserted_count = 0
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                
                # Format vectors for Pinecone
                formatted_batch = []
                for vid, embedding, metadata in batch:
                    # Ensure metadata is JSON-serializable
                    clean_metadata = self._clean_metadata(metadata)
                    
                    formatted_batch.append({
                        "id": str(vid),
                        "values": embedding,
                        "metadata": clean_metadata
                    })
                
                # Upsert batch
                upsert_response = self.index.upsert(
                    vectors=formatted_batch,
                    namespace=self. namespace
                )
                
                upserted_count += len(batch)
                logger.info(f"Upserted batch {i//batch_size + 1}: {len(batch)} vectors (Total: {upserted_count}/{len(vectors)})")
            
            logger.info(f"Successfully upserted {upserted_count} vectors to Pinecone")
            return upserted_count
        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            raise
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata to be JSON-serializable for Pinecone
        
        Args:
            metadata: Raw metadata
            
        Returns:
            Cleaned metadata
        """
        cleaned = {}
        
        for key, value in metadata. items():
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, list):
                # Convert list to comma-separated string if it's a list of strings
                if value and isinstance(value[0], str):
                    cleaned[key] = ", ".join(value)
                else:
                    cleaned[key] = str(value)
            elif isinstance(value, dict):
                cleaned[key] = json.dumps(value)
            elif value is None:
                cleaned[key] = "null"
            else:
                cleaned[key] = str(value)
        
        return cleaned
    
    # ===================== QUERY OPERATIONS =====================
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def query_vectors(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query similar vectors from Pinecone
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            filter_dict: Optional metadata filter (Pinecone filter syntax)
            include_metadata: Whether to include metadata
            
        Returns:
            List of matching vectors with scores and metadata
        """
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=include_metadata,
                filter=filter_dict
            )
            
            formatted_results = []
            for match in results. get("matches", []):
                formatted_results.append({
                    "id": match. get("id"),
                    "score": match.get("score"),
                    "metadata": match. get("metadata", {})
                })
            
            logger.info(f"Found {len(formatted_results)} matching vectors (top_k={top_k})")
            return formatted_results
        except Exception as e:
            logger. error(f"Error querying vectors: {str(e)}")
            raise
    
    # ===================== HYBRID SEARCH OPERATIONS =====================
    
    def hybrid_search(
        self,
        query_embedding: List[float],
        job_data: Dict[str, Any],
        top_k: int = 5,
        skill_filters: Optional[List[str]] = None,
        industry_filter: Optional[str] = None,
        min_similarity_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic similarity and metadata filtering
        
        Smart retrieval strategy:
        1.  Semantic search based on embedding similarity
        2. Filter by skills (if provided)
        3. Filter by industry (if provided)
        4. Re-rank by relevance score
        5. Return top results with similarity confidence
        
        Args:
            query_embedding: Query embedding
            job_data: Job metadata for filtering
            top_k: Number of results
            skill_filters: List of skills to filter by
            industry_filter: Industry to filter by
            min_similarity_score: Minimum similarity score threshold
            
        Returns:
            Ranked list of relevant past projects
        """
        try:
            # Build Pinecone filter
            filters = {}
            
            if skill_filters:
                # Filter for chunks that have at least one matching skill
                filters["skills"] = {"$in": skill_filters}
            
            if industry_filter:
                filters["industry"] = {"$eq": industry_filter}
            
            logger.info(f"Performing hybrid search with filters: {filters}")
            
            # Query Pinecone with filters (fetch more to account for filtering)
            results = self.query_vectors(
                query_embedding=query_embedding,
                top_k=top_k * 3,  # Get more results for filtering
                filter_dict=filters if filters else None,
                include_metadata=True
            )
            
            # Filter by minimum similarity score
            filtered_results = [r for r in results if r["score"] >= min_similarity_score]
            
            # Re-rank by relevance score
            sorted_results = sorted(filtered_results, key=lambda x: x["score"], reverse=True)
            
            # Return top k
            final_results = sorted_results[:top_k]
            
            logger.info(f"Hybrid search returned {len(final_results)} results (score threshold: {min_similarity_score})")
            return final_results
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return []
    
    def smart_search(
        self,
        query_embedding: List[float],
        job_description: str,
        company_name: str,
        required_skills: List[str],
        industry: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Advanced smart search combining multiple strategies for best results
        
        Strategy:
        1. Primary search: By skills and industry
        2. Secondary search: By company and job description similarity
        3. Tertiary search: Generic similarity search
        4.  Combine and rank all results
        
        Args:
            query_embedding: Query embedding
            job_description: Job description
            company_name: Company name
            required_skills: Required skills
            industry: Industry
            top_k: Number of results
            
        Returns:
            Dictionary with primary, secondary, and combined results
        """
        try:
            results = {
                "primary_results": [],  # Skills + Industry match
                "secondary_results": [],  # Similar companies
                "all_results": [],
                "search_metadata": {
                    "query_skills": required_skills,
                    "query_industry": industry,
                    "query_company": company_name
                }
            }
            
            # Primary search: Skills + Industry
            primary = self.hybrid_search(
                query_embedding=query_embedding,
                job_data={"industry": industry},
                top_k=top_k,
                skill_filters=required_skills if required_skills else None,
                industry_filter=industry,
                min_similarity_score=0.6
            )
            results["primary_results"] = primary
            
            # Generic similarity search
            generic = self.query_vectors(
                query_embedding=query_embedding,
                top_k=top_k * 2,
                include_metadata=True
            )
            results["secondary_results"] = generic
            
            # Combine and deduplicate
            seen_ids = set()
            combined = []
            
            for result in results["primary_results"]:
                vid = result["id"]
                if vid not in seen_ids:
                    combined.append(result)
                    seen_ids.add(vid)
            
            for result in results["secondary_results"]:
                vid = result["id"]
                if vid not in seen_ids:
                    combined.append(result)
                    seen_ids. add(vid)
            
            results["all_results"] = combined[:top_k]
            logger.info(f"Smart search completed with {len(results['all_results'])} combined results")
            
            return results
        except Exception as e:
            logger.error(f"Error in smart search: {str(e)}")
            return {"all_results": [], "error": str(e)}
    
    # ===================== DELETE OPERATIONS =====================
    
    def delete_vectors(self, ids: List[str]) -> int:
        """
        Delete vectors from Pinecone
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            Number of vectors deleted
        """
        try:
            delete_response = self.index.delete(
                ids=[str(vid) for vid in ids],
                namespace=self.namespace
            )
            logger.info(f"Deleted {len(ids)} vectors from Pinecone")
            return len(ids)
        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")
            raise
    
    def delete_by_contract(self, contract_id: str) -> int:
        """
        Delete all vectors for a specific contract
        
        Args:
            contract_id: Contract ID to delete
            
        Returns:
            Number of vectors deleted
        """
        try:
            # Query all vectors for this contract first
            response = self.index.query(
                vector=[0.0] * self.dimension,  # Dummy query
                top_k=10000,
                namespace=self.namespace,
                include_metadata=True,
                filter={"contract_id": {"$eq": contract_id}}
            )
            
            ids_to_delete = [match["id"] for match in response. get("matches", [])]
            
            if ids_to_delete:
                return self.delete_vectors(ids_to_delete)
            
            logger.info(f"No vectors found for contract: {contract_id}")
            return 0
        except Exception as e:
            logger.error(f"Error deleting by contract: {str(e)}")
            return 0
    
    # ===================== MANAGEMENT OPERATIONS =====================
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        try:
            stats = self.index.describe_index_stats()
            logger. info(f"Index stats retrieved: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {}
    
    def get_namespace_stats(self) -> Dict[str, Any]:
        """Get statistics for the current namespace"""
        try:
            stats = self.index.describe_index_stats()
            namespaces = stats.get("namespaces", {})
            namespace_stats = namespaces.get(self.namespace, {})
            
            return {
                "namespace": self. namespace,
                "vector_count": namespace_stats.get("vector_count", 0),
                "index_fullness": stats.get("index_fullness", 0)
            }
        except Exception as e:
            logger.error(f"Error getting namespace stats: {str(e)}")
            return {}
    
    def clear_namespace(self) -> bool:
        """Clear all vectors from the namespace"""
        try:
            self. index.delete(delete_all=True, namespace=self.namespace)
            logger.info(f"Cleared namespace: {self.namespace}")
            return True
        except Exception as e:
            logger.error(f"Error clearing namespace: {str(e)}")
            return False
    
    def health_check(self) -> bool:
        """Check if Pinecone index is healthy"""
        try:
            stats = self.get_index_stats()
            logger.info("Pinecone health check: OK")
            return bool(stats)
        except Exception as e:
            logger.error(f"Pinecone health check failed: {str(e)}")
            return False