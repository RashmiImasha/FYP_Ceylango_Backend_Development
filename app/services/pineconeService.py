from pinecone import Pinecone, ServerlessSpec
import torch, io, requests, logging
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List, Dict
from app.config.settings import settings
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class PineconeService:  

    def __init__(self):
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME

        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=512,  
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        self.index = self.pc.Index(self.index_name)
        self.genai_client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.clip_model = None
        self.clip_processor = None


    
    def _load_clip_model(self):
        if self.clip_model is None:
            logger.info("Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained(settings.CLIP_MODEL)
            self.clip_processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL)
            self.clip_model.eval()

        
    

    def generate_text_embedding(self, text: str) -> List[float]:

        try:
            response = self.genai_client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=types.EmbedContentConfig(
                    task_type='RETRIEVAL_DOCUMENT',
                    output_dimensionality=512
                )                    
            )      
            embedding = response.embeddings[0].values
        
            if embedding is None or len(embedding) != 512:
                raise ValueError(f"Invalid embedding dimension: {len(embedding) if embedding else 0}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating text embedding: {str(e)}")
            raise
    
    def generate_image_embedding(self, image_source) -> List[float]:
        
        self._load_clip_model()        
        try:
            image = None
            
            # Handle different input types
            if isinstance(image_source, str):   # URL
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))
                
            elif isinstance(image_source, bytes):   # Bytes
                image = Image.open(io.BytesIO(image_source))
                
            elif isinstance(image_source, Image.Image):   # PIL Image
                image = image_source
                
            else:
                raise ValueError(f"Unsupported type: {type(image_source)}")
            
            # Validate image
            if image is None:
                raise ValueError("Failed to create image object")
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            logger.info(f"Image ready for CLIP: {image.size}, mode={image.mode}")
            
            # OPTIMIZATION 2: Process with CLIP on optimized image
            inputs = self.clip_processor(images=image, return_tensors="pt")
            
            # OPTIMIZATION 3: Use torch.no_grad() for inference (no gradient computation)
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            # Normalize embedding
            embedding = image_features / image_features.norm(dim=-1, keepdim=True)
            return embedding[0].cpu().numpy().tolist()

        except Exception as e:
            logger.error(f"Embedding generation error: {str(e)}", exc_info=True)
            raise       
    
         
    def upsert_destination_image(self, destination_id: str, destination_data: Dict) -> bool:
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
                namespace='destinationImages'
            )

            return True

        except Exception as e:
            raise

    
    def upsert_destination_text(self, destination_id: str, destination_data: Dict) -> bool:
        
        try:
            # Combine all text fields for embedding
            text_to_embed = f"""
                {destination_data.get('destination_name', '')}
                {destination_data.get('description', '')[:2000]}
                Category: {destination_data.get('category_name', '')}
                District: {destination_data.get('district_name', '')}
                Location: {destination_data.get('latitude', 0)}, {destination_data.get('longitude', 0)}""".strip()
            
            if not text_to_embed or len(text_to_embed.strip()) < 10:
                logger.warning(f"Insufficient text for destination {destination_id}")
                return False
                            
            # Generate text embedding 
            embedding = self.generate_text_embedding(text_to_embed)
          
            metadata = {
                'destination_id': destination_id,
                'destination_name': destination_data.get('destination_name', '')[:200],   
                'district_name': destination_data.get('district_name', ''),  
                'category_name': destination_data.get('category_name', '')             
            }
            
            # Upsert to Pinecone 
            self.index.upsert(
                vectors=[{
                    'id': destination_id,  
                    'values': embedding,
                    'metadata': metadata
                }],
                namespace='destinationTextdata' 
            )

            return True

        except Exception as e:
            raise
    
    def delete_destination_image(self, destination_id: str) -> bool:
        try:
            # Delete all images for this destination
            ids_to_delete = [f"{destination_id}_{i}" for i in range(10)]  # adjust max count if needed
            self.index.delete(ids=ids_to_delete, namespace='destinationImages')
            return True
        except Exception as e:
            raise
    
    def delete_destination_text(self, destination_id: str) -> bool:
        """
        Delete destination text embedding from Pinecone
        """
        try:
            self.index.delete(
                ids=[destination_id],  # Single ID 
                namespace='destinationTextdata'
            )
            return True
        except Exception as e:
            raise

    

_pinecone_service = None

def get_pinecone_service() -> PineconeService:    
    global _pinecone_service
    if _pinecone_service is None:
        _pinecone_service = PineconeService()
    return _pinecone_service

