from pinecone import Pinecone, ServerlessSpec
from transformers import CLIPProcessor, CLIPModel
import torch, io, requests, logging
from PIL import Image
from typing import List, Dict, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config.settings import settings

logger = logging.getLogger(__name__)

class PineconeService:  

    def __init__(self):
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = "sri-lanka-locations"

        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=512,  # CLIP ViT-B/32 produces 512-dim embeddings
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        self.index = self.pc.Index(self.index_name)

        # initialize CLIP model for image embeddings
        logger.info("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()

        # initialize Google embedding for text 
        self.text_embedder = GoogleGenerativeAIEmbeddings(
            model="gemini-2.5-flash-lite-embedding-001",
            google_api_key=settings.GOOGLE_API_KEY
        )
        
        logger.info("PineconeService initialized successfully...")

    def generate_image_embedding(self, image_source) -> List[float]:
        """
        Generate CLIP embedding from image        
        Args:
            image_source: PIL Image, bytes, or URL string            
        Returns:
            List of floats representing the embedding vector
        """
        
        try:
            # logger.info(f"=== generate_image_embedding START ===")
            # logger.info(f"image_source type: {type(image_source)}")
            # logger.info(f"image_source is None: {image_source is None}")
            
            # Validate input is not None FIRST
            if image_source is None:
                logger.error("generate_image_embedding received None as image_source")
                raise ValueError("Image source cannot be None")
            
            image = None
            
            if isinstance(image_source, str):   # URL
                logger.info(f"Processing as URL: {image_source[:100]}")
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert('RGB')
                
            elif isinstance(image_source, bytes):   # Bytes
                logger.info(f"Processing as bytes: {len(image_source)} bytes")
                image = Image.open(io.BytesIO(image_source)).convert('RGB')
                
            elif isinstance(image_source, Image.Image):   # PIL Image
                logger.info(f"Processing as PIL Image")
                # logger.info(f"Input image size: {image_source.size}, mode: {image_source.mode}")
                image = image_source.convert('RGB')
                # logger.info(f"After convert: image is None={image is None}")
                
            else:
                logger.error(f"Unsupported type: {type(image_source)}")
                raise ValueError(f"Unsupported image source type: {type(image_source)}")
            
            # Verify image was successfully created
            if image is None:
                logger.error("Image is None after processing")
                raise ValueError("Failed to create image object from source")
            
            # logger.info(f"Image created successfully: size={image.size}, mode={image.mode}")
            
            # Process image with CLIP
            # logger.info("Processing with CLIP...")
            inputs = self.clip_processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            # Normalize embedding
            embedding = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logger.info(f"Embedding generated successfully, shape: {embedding.shape}")
            return embedding[0].cpu().numpy().tolist()

        except Exception as e:
            logger.error(f"Error in generate_image_embedding: {str(e)}", exc_info=True)
            raise         
        
    def upsert_destination(self, destination_id: str, destination_data: Dict) -> bool:
        """
        Add or update destination in Pinecone using ALL images + lat/lon for embedding and metadata
        """
        try:
            image_urls = destination_data.get('destination_image', [])
            if not image_urls:
                raise ValueError("No image URLs found for destination")

            # Generate embeddings for all images
            vectors_to_upsert = []

            for idx, img_url in enumerate(image_urls):
                # print(f"Generating embedding for image {idx + 1}/{len(image_urls)}: {img_url}")
                embedding = self.generate_image_embedding(img_url)
                
                # Prepare metadata for each image
                metadata = {
                    'destination_id': destination_id,
                    'name': destination_data.get('destination_name', '')[:200],
                    'latitude': float(destination_data.get('latitude', 0)),
                    'longitude': float(destination_data.get('longitude', 0)),
                    'district': destination_data.get('district_name', '')[:100],
                    'category': destination_data.get('category_name', '')[:100],
                    'description': destination_data.get('description', '')[:500],
                    'image_url': img_url 
                }
                
                vectors_to_upsert.append({
                    'id': f"{destination_id}_{idx}",  # unique ID per image
                    'values': embedding,
                    'metadata': metadata
                })

            # Upsert all embeddings to Pinecone
            self.index.upsert(
                vectors=vectors_to_upsert,
                namespace='destinations'
            )

            print(f"Successfully upserted {len(image_urls)} images for destination {destination_id} to Pinecone...")
            return True

        except Exception as e:
            logger.error(f"Error upserting to Pinecone: {str(e)}")
            raise

    def delete_destination(self, destination_id: str) -> bool:
        try:
            # Delete all images for this destination
            ids_to_delete = [f"{destination_id}_{i}" for i in range(10)]  # adjust max count if needed
            self.index.delete(ids=ids_to_delete, namespace='destinations')
            print(f"Deleted destination {destination_id} from Pinecone")
            return True
        except Exception as e:
            logger.error(f"Error deleting from Pinecone: {str(e)}")
            raise
    
    def search_similar_by_image(
        self,
        image_source,
        gps_location: Optional[Dict] = None,
        top_k: int = 5,
        radius_km: float = 10.0
    ) -> List[Dict]:
        """
        Search for similar locations using image
        
        Args:
            image_source: Image (PIL, bytes, or URL)
            gps_location: Optional dict with 'lat' and 'lng' keys
            top_k: Number of results to return
            radius_km: Search radius in kilometers
            
        Returns:
            List of matching destinations with scores
        """
        try:
            # Generate image embedding
            embedding = self.generate_image_embedding(image_source)
            
            # Prepare filter for GPS if provided
            filter_dict = None
            if gps_location:
                # Convert km to degrees (approximate: 1 degree ≈ 111 km)
                lat_range = radius_km / 111.0
                lng_range = radius_km / (111.0 * abs(gps_location['lat']))
                
                filter_dict = {
                    "$and": [
                        {"latitude": {"$gte": gps_location['lat'] - lat_range}},
                        {"latitude": {"$lte": gps_location['lat'] + lat_range}},
                        {"longitude": {"$gte": gps_location['lng'] - lng_range}},
                        {"longitude": {"$lte": gps_location['lng'] + lng_range}}
                    ]
                }
            
            # Query Pinecone
            results = self.index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
                namespace='destinations',
                filter=filter_dict
            )
            
            # Format results
            matches = []
            for match in results.get('matches', []):
                matches.append({
                    'id': match['id'],
                    'score': float(match['score']),
                    'metadata': match.get('metadata', {})
                })
            
            return matches
        
        except Exception as e:
            logger.error(f"Error searching Pinecone: {str(e)}")
            raise
    
    def search_similar_by_text(
        self,
        query: str,
        gps_location: Optional[Dict] = None,
        top_k: int = 5,
        radius_km: float = 10.0
    ) -> List[Dict]:
        """
        Search for similar locations using text query
        
        Args:
            query: Search query text
            gps_location: Optional dict with 'lat' and 'lng' keys
            top_k: Number of results to return
            radius_km: Search radius in kilometers
            
        Returns:
            List of matching destinations with scores
        """
        try:
            # Generate text embedding
            embedding = self.generate_text_embedding(query)
            
            # Prepare filter for GPS if provided
            filter_dict = None
            if gps_location:
                lat_range = radius_km / 111.0
                lng_range = radius_km / (111.0 * abs(gps_location['lat']))
                
                filter_dict = {
                    "$and": [
                        {"latitude": {"$gte": gps_location['lat'] - lat_range}},
                        {"latitude": {"$lte": gps_location['lat'] + lat_range}},
                        {"longitude": {"$gte": gps_location['lng'] - lng_range}},
                        {"longitude": {"$lte": gps_location['lng'] + lng_range}}
                    ]
                }
            
            # Query Pinecone
            results = self.index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
                namespace='destinations',
                filter=filter_dict
            )
            
            # Format results
            matches = []
            for match in results.get('matches', []):
                matches.append({
                    'id': match['id'],
                    'score': float(match['score']),
                    'metadata': match.get('metadata', {})
                })
            
            return matches
        
        except Exception as e:
            logger.error(f"Error searching Pinecone: {str(e)}")
            raise

_pinecone_service = None

def get_pinecone_service() -> PineconeService:    
    global _pinecone_service
    if _pinecone_service is None:
        _pinecone_service = PineconeService()
    return _pinecone_service

# def generate_text_embedding(self, text: str) -> List[float]:
    #     """
    #     Generate embedding from text using Google Gemini
        
    #     Args:
    #         text: Text to embed
            
    #     Returns:
    #         List of floats representing the embedding vector
    #     """
    #     try:
    #         embedding = self.text_embedder.embed_query(text)
            
    #         # Pad or truncate to 512 dimensions to match CLIP
    #         if len(embedding) < 512:
    #             embedding = embedding + [0.0] * (512 - len(embedding))
    #         elif len(embedding) > 512:
    #             embedding = embedding[:512]
            
    #         return embedding
        
    #     except Exception as e:
    #         logger.error(f"Error generating text embedding: {str(e)}")
    #         raise

    # def generate_multimodal_embedding(
    #     self, 
    #     text: str, 
    #     image_urls: List[str],
    #     text_weight: float = 0.3,
    #     image_weight: float = 0.7
    # ) -> List[float]:
    #     """
    #     Generate combined embedding from text and images
        
    #     Args:
    #         text: Text description
    #         image_urls: List of image URLs
    #         text_weight: Weight for text embedding (default 0.3)
    #         image_weight: Weight for image embeddings (default 0.7)
            
    #     Returns:
    #         Combined embedding vector
    #     """
    #     try:
    #         # Generate text embedding
    #         text_emb = self.generate_text_embedding(text)
    #         text_vector = torch.tensor(text_emb)
            
    #         # Generate and average image embeddings
    #         if image_urls:
    #             image_embeddings = []
    #             for url in image_urls[:3]:  # Limit to first 3 images for performance
    #                 try:
    #                     img_emb = self.generate_image_embedding(url)
    #                     image_embeddings.append(torch.tensor(img_emb))
    #                 except Exception as e:
    #                     logger.warning(f"Failed to process image {url}: {str(e)}")
    #                     continue
                
    #             if image_embeddings:
    #                 # Average all image embeddings
    #                 avg_image_vector = torch.stack(image_embeddings).mean(dim=0)
                    
    #                 # Combine text and image with weights
    #                 combined = (text_weight * text_vector + image_weight * avg_image_vector)
                    
    #                 # Normalize
    #                 combined = combined / combined.norm()
                    
    #                 return combined.tolist()
            
    #         # If no images, return normalized text embedding
    #         text_vector = text_vector / text_vector.norm()
    #         return text_vector.tolist()
        
    #     except Exception as e:
    #         logger.error(f"Error generating multimodal embedding: {str(e)}")
    #         raise

# import torch
# print(torch.__version__)
# print("CUDA available:", torch.cuda.is_available())

# model="gemini-embedding-001",