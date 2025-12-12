from app.services.pineconeService import get_pinecone_service
from app.config.settings import settings
import google.generativeai as genai
from PIL import Image
import logging, re, json, asyncio
from typing import Dict, Tuple
from geopy.geocoders import Nominatim
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class LocationIdentificationAgent:
    
    def __init__(self):
        
        self.pinecone_service = get_pinecone_service()
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.text_model = genai.GenerativeModel(settings.GEMINI_MODEL)

        # Thread pool for CPU-bound operations
        self._executor = ThreadPoolExecutor(max_workers=10)
        
    def _resize_image(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        """
        Resize image to max dimension and convert to RGB
        """
        try:
            width, height = image.size

            if max(width, height) > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Image resized from {width}x{height} to {new_width}x{new_height}")
            
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            return image
        except Exception as e:
            logger.error(f"Error optimizing image: {str(e)}")
            raise
    
    async def _resize_image_async(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._resize_image, image, max_size)
    
    async def prepare_image_and_embedding(self, image: Image.Image, gps_location: Dict[str, float]) -> Tuple[Image.Image, list, Dict[str, float]]:
        if image is None:
            raise ValueError("Image cannot be None")
        if not gps_location or 'lat' not in gps_location or 'lng' not in gps_location:
            gps_location = {'lat': 0.0, 'lng': 0.0}

        loop = asyncio.get_event_loop()
        resized_image = await self._resize_image_async(image)
        embedding = await loop.run_in_executor(self._executor, self.pinecone_service.generate_image_embedding, resized_image) 

        return resized_image, embedding, gps_location
    
    async def _analyze_image(self, image: Image.Image, gps_location: Dict[str, float]) -> str:
        
        try:
            
            prompt = f"""
                Analyze this image captured in Sri Lanka at GPS: {gps_location.get('lat', 'unknown')}, {gps_location.get('lng', 'unknown')}.

                Provide structured analysis:
                1. **Main Subject**: Primary landmark/location (be specific)
                2. **Architecture**: Buildings, monuments, structures (Buddhist stupa, colonial building, etc.)
                3. **Natural Features**: Landscape (mountains, water, vegetation, beaches)
                4. **Cultural Elements**: Symbols, statues, inscriptions, religious markers
                5. **Distinctive Features**: Unique identifiers for exact location
                6. **Era**: Historical period (ancient, colonial, modern)
                7. **Category**: Temple/Beach/Mountain/Fort/National Park/City/Village/etc.
                8. **Visible Text**: Signs, boards, inscriptions (crucial for ID)

                CRITICAL:
                - Describe ONLY what you SEE
                - NO location guessing from GPS
                - NO temple names unless visible in text
                - Objective visual features only

                Be detailed about VISIBLE features.
                """
            
            response = self.text_model.generate_content([prompt, image])            
            return response.text
            
        except Exception as e:
            logger.error(f"Visual analysis error: {str(e)}")
            return f"Error: {str(e)}"
    
    async def _search_database(self, embedding: list, gps_location: Dict[str, float]) -> Tuple[list, str]:
        """Search database using embeddings (async wrapper)"""
        try:
            if embedding is None:
                return [], "Error: No embedding available"
            
            # Prepare filter
            filter_dict = {}
            if gps_location:
                lat, lng = gps_location.get('lat', 0), gps_location.get('lng', 0)
                radius_km = 10.0
                lat_range = radius_km / 111.0
                lng_range = radius_km / (111.0 * abs(lat)) if lat != 0 else radius_km / 111.0
                filter_dict = {
                    "$and": [
                        {"latitude": {"$gte": lat - lat_range, "$lte": lat + lat_range}},
                        {"longitude": {"$gte": lng - lng_range, "$lte": lng + lng_range}},
                    ]
                }
            
            # Query Pinecone asynchronously
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self._executor,
                lambda: self.pinecone_service.index.query(
                    vector=embedding,
                    top_k=3,  
                    include_metadata=True,
                    namespace='destinationImages',
                    filter=filter_dict
                )
            )
            
            matches = results.get("matches", [])
            if not matches:
                return [], "No matches found within 10km"
            
            # Format results
            output_lines = [f"Found {len(matches)} matches:\n"]
            for i, match in enumerate(matches, 1):
                score = match['score']
                meta = match['metadata']
                confidence = "HIGH" if score > 0.85 else "MEDIUM" if score > 0.75 else "LOW"
                
                output_lines.append(
                    f"{i}. {meta.get('name', 'Unknown')} [{confidence}: {score:.2%}]\n"
                    f"   District: {meta.get('district', 'Unknown')}\n"
                    f"   Category: {meta.get('category', 'Unknown')}\n"
                    f"   Description: {meta.get('description', 'N/A')}\n"
                )
            
            return matches, "".join(output_lines)
            
        except Exception as e:
            logger.error(f"Database search error: {str(e)}")
            return [], f"Error: {str(e)}"
    
    async def _generate_content_with_gemini(
        self,
        destination_name: str,
        district_name: str,
        category_name: str,
        db_description: str,
        visual_analysis: str,
        gps_location: dict,
        confidence: str
    ) -> dict:
        
        try:
            prompt = f"""You are an expert Sri Lankan tourism guide.

            A tourist captured **{destination_name}** in Sri Lanka.

            CONFIRMED INFO:
            - Location: {destination_name}
            - District: {district_name}
            - Category: {category_name}
            - Confidence: {confidence}
            - GPS: {gps_location.get('lat')}, {gps_location.get('lng')}

            DATABASE: {db_description if db_description else 'Limited info'}
            VISUAL: {visual_analysis[:600]}

            Generate JSON (no markdown):
            {{
            "historical_background": "history, who built it, events",
            "cultural_significance": "religious/cultural importance, heritage",
            "what_makes_it_special": "unique features, architectural details",
            "visitor_experience": "best time, what to see/do, tips",
            "interesting_facts": ["fact 1", "fact 2", "fact 3"]
            }}

            Use your knowledge of {destination_name}. Be accurate, enthusiastic. Return ONLY JSON.
            """
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self._executor,
                lambda: self.text_model.generate_content(prompt)
            )
            generated_text = response.text.strip()
            generated_text = re.sub(r'^```json\s*|\s*```$', '', generated_text, flags=re.MULTILINE).strip()
            
            content_dict = json.loads(generated_text)
            
            # Ensure text fields are strings
            for field in ['historical_background', 'cultural_significance', 'what_makes_it_special', 'visitor_experience']:
                value = content_dict.get(field, "")
                if isinstance(value, list):
                    content_dict[field] = " ".join(value)
            
            # Ensure facts is a list
            facts = content_dict.get('interesting_facts', [])
            if not isinstance(facts, list):
                content_dict['interesting_facts'] = [str(facts)]
            
            return content_dict
            
        except json.JSONDecodeError:
            logger.error("JSON parsing failed")
            return self._get_generated_content(destination_name, district_name)
        except Exception as e:
            logger.error(f"Content generation error: {str(e)}")
            return self._get_generated_content(destination_name, district_name)
    
    def _get_generated_content(self, dest_name: str, district: str) -> dict:
        
        return {
            "historical_background": f"{dest_name} is a notable site in {district}, Sri Lanka.",
            "cultural_significance": "This location offers insight into Sri Lankan heritage.",
            "what_makes_it_special": "A remarkable destination worth visiting.",
            "visitor_experience": "Visit during daylight for best experience.",
            "interesting_facts": [
                "Important cultural site in Sri Lanka.",
                "Attracts local and international visitors.",
                "Photography usually permitted."
            ]
        }
    
    async def _reverse_geocode_location(self, lat: float, lng: float) -> dict:
        """Async reverse geocoding"""
        try:
            loop = asyncio.get_event_loop()
            geolocator = Nominatim(user_agent="ceylango_agent")
            
            location = await loop.run_in_executor(
                self._executor,
                lambda: geolocator.reverse((lat, lng), language="en", timeout=10)
            )
            
            if location and location.raw.get("address"):
                address = location.raw["address"]
                
                district_name = (
                    address.get("state_district") or 
                    address.get("county") or 
                    address.get("state")
                )
                city_name = (
                    address.get("city") or 
                    address.get("town") or 
                    address.get("suburb")
                )
                village_name = (
                    address.get("village") or 
                    address.get("hamlet")
                )
                
                geocoded_landmark_name = (
                    address.get("amenity") or
                    address.get("tourism") or
                    address.get("attraction") or
                    address.get("historic") or
                    address.get("building") or
                    address.get("monument") or
                    address.get("university") or  # Add this
                    address.get("college") or      # Add this
                    address.get("school") or       # Add this
                    address.get("house_name") or   # Add this
                    address.get("road") 
                )
                
                logger.info(f"Geocoded landmark name: {geocoded_landmark_name}")
                
                return {
                    "district_name": district_name,
                    "city_name": city_name,
                    "village_name": village_name,
                    "geocoded_landmark_name": geocoded_landmark_name,
                    "full_address": address
                }
            
            return {}
            
        except Exception as e:
            logger.warning(f"Geocoding failed: {str(e)}")
            return {}

    async def _web_search(self, query: str, visual_features: str = None, gps_location: dict = None) -> dict:
        """
        Identify location using reverse geocoding + Gemini with robust error handling
        """
        try:
            lat = gps_location.get("lat") or gps_location.get("latitude") if gps_location else None
            lng = gps_location.get("lng") or gps_location.get("longitude") if gps_location else None
            
            geocode_result = {}
            if lat and lng:
                geocode_result = await self._reverse_geocode_location(lat, lng)
            
            district_name = geocode_result.get("district_name")
            city_name = geocode_result.get("city_name")
            village_name = geocode_result.get("village_name")
            geocoded_landmark_name = geocode_result.get("geocoded_landmark_name")

            # Build context from available location data
            location_context = ", ".join(filter(None, [
                village_name,
                city_name, 
                district_name
            ])) or "Unknown region in Sri Lanka"

            # Use geocoded name if available
            if geocoded_landmark_name:
                logger.info(f"Using geocoded name directly: {geocoded_landmark_name}")
                
                prompt = f"""You are a Sri Lankan tourism expert.

                    A tourist has captured an image of **{geocoded_landmark_name}** in {district_name or 'Sri Lanka'}.

                    CONFIRMED INFORMATION:
                    - Location Name: {geocoded_landmark_name}
                    - District: {district_name or 'Unknown'}
                    - Area: {location_context}
                    - GPS: {lat}, {lng}

                    Generate tourist information for this location.

                    Respond with ONLY a valid JSON object (no markdown, no preamble):

                    {{
                    "destination_name": "{geocoded_landmark_name}",
                    "category": "Temple/Beach/Fort/Mountain/National Park/City/Village/Ancient City/Others",
                    "historical_background": "history and background",
                    "cultural_significance": "cultural importance",
                    "what_makes_it_special": "unique features",
                    "visitor_experience": "visiting tips",
                    "interesting_facts": ["fact 1", "fact 2", "fact 3"]
                    }}
                    """
            
            else:
                # Fallback with better prompt
                logger.warning("No geocoded name found - asking Gemini to identify")
                
                prompt = f"""You are a Sri Lankan tourism AI assistant.

                    LOCATION CONTEXT:
                    - GPS: {lat or 'unknown'}, {lng or 'unknown'}
                    - Region: {location_context}
                    - District: {district_name or 'unknown'}
                    - Village: {village_name or 'unknown'}

                    VISUAL ANALYSIS:
                    {visual_features if visual_features else 'No visual data provided'}

                    TASK:
                    Based on the GPS location in {district_name or 'this district'}, identify the most likely landmark or temple in this area.

                    CRITICAL RULES:
                    1. The landmark MUST be in {district_name or 'this district'}
                    2. If you cannot identify with confidence, use a descriptive name like "Buddhist Temple in {village_name or district_name}"
                    3. NEVER invent a name - if uncertain, be descriptive

                    Respond with ONLY a valid JSON object (no markdown, no text before or after):

                    {{
                    "destination_name": "name of the location",
                    "category": "Temple/Beach/Fort/Mountain/National Park/City/Village/Ancient City/Others",
                    "historical_background": "historical information",
                    "cultural_significance": "cultural importance",
                    "what_makes_it_special": "unique features",
                    "visitor_experience": "visiting tips",
                    "interesting_facts": ["fact 1", "fact 2", "fact 3"]
                    }}

                    Generate the JSON now:"""

            # ✅ FIX 3: Robust Gemini call with error handling
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    self._executor,
                    lambda: self.text_model.generate_content(prompt)
                )
                
                if not response or not response.text:
                    raise ValueError("Gemini returned empty response")
                
                generated_text = response.text.strip()
                # logger.info(f"Gemini raw response (first 200 chars): {generated_text[:200]}")
                
                # Remove markdown fences
                generated_text = re.sub(r'^```json\s*|\s*```$', '', generated_text, flags=re.MULTILINE).strip()
                
                # ✅ FIX 4: Validate JSON before parsing
                if not generated_text or not generated_text.startswith('{'):
                    json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
                    if json_match:
                        generated_text = json_match.group(0)
                    else:
                        raise ValueError("No JSON object found in response")
                
                content_dict = json.loads(generated_text)
                
            except (json.JSONDecodeError, Exception) as e:
                logger.error(f"gemini generation/parsing failed: {str(e)}")
                return self._create_fallback_content(
                    geocoded_landmark_name or f"Location in {village_name or district_name or 'Sri Lanka'}",
                    district_name,
                    village_name,
                    location_context
                )
                       
            return {
                "destination_name": content_dict.get("destination_name", geocoded_landmark_name or "Unknown Location"),
                "district_name": district_name or content_dict.get("district_name", "Unknown"),
                "category": content_dict.get("category", "Others"),
                "historical_background": content_dict.get("historical_background", ""),
                "cultural_significance": content_dict.get("cultural_significance", ""),
                "what_makes_it_special": content_dict.get("what_makes_it_special", ""),
                "visitor_experience": content_dict.get("visitor_experience", ""),
                "interesting_facts": content_dict.get("interesting_facts", [])
            }

        except Exception as e:
            logger.error(f"Web search tool error: {str(e)}", exc_info=True)
            return self._create_fallback_content("Unknown Location", "Unknown", None, "Unknown region")

    def _create_fallback_content(self, destination_name: str, district_name: str, 
                                village_name: str = None, location_context: str = "Unknown") -> dict:
        """
        Create generic fallback content when Gemini fails
        """
        return {
            "destination_name": destination_name,
            "district_name": district_name or "Unknown",
            "category": "Others",
            "historical_background": f"This location is situated in {location_context}. More detailed historical information may become available as our database expands.",
            "cultural_significance": f"As part of {district_name or 'Sri Lanka'}'s heritage, this site contributes to the cultural landscape of the region.",
            "what_makes_it_special": "This location offers visitors an authentic Sri Lankan experience and a glimpse into local culture.",
            "visitor_experience": "When visiting, be respectful of local customs and traditions. Consider hiring a local guide for deeper insights into the area.",
            "interesting_facts": [
                f"Located in {district_name or 'Sri Lanka'}",
                f"Part of the {village_name or 'local'} community",
                "Further details available from local tourism offices"
            ]
        }

    async def identify_and_generate_content(self, image: Image.Image, gps_location: Dict[str, float]) -> Dict:
        """
        Main entry point - orchestrates workflow with parallel processing
        """
        try:
            # Step 1: Prepare image and embedding (async)
            resized_image, embedding, gps_location = await self.prepare_image_and_embedding(image, gps_location)
            
            # Step 2: Run visual analysis and database search IN PARALLEL
            visual_task = asyncio.create_task(self._analyze_image(resized_image, gps_location))
            db_task = asyncio.create_task(self._search_database(embedding, gps_location))
            
            visual_analysis, (db_matches, db_output) = await asyncio.gather(visual_task, db_task)
                        
            # Step 3: Determine confidence and decide next steps
            if db_matches:
                top_match = db_matches[0]
                confidence_score = top_match['score']
                
                if confidence_score > 0.75:
                    # HIGH confidence - use database result
                    meta = top_match['metadata']
                    
                    content_dict = await self._generate_content_with_gemini(
                        destination_name=meta.get('name', 'Unknown'),
                        district_name=meta.get('district', 'Unknown'),
                        category_name=meta.get('category', 'Others'),
                        db_description=meta.get('description', ''),
                        visual_analysis=visual_analysis,
                        gps_location=gps_location,
                        confidence="High"
                    )
                    
                    return {
                        "success": True,
                        "destination_name": meta.get('name', 'Unknown'),
                        "district_name": meta.get('district', 'Unknown'),
                        "category": meta.get('category', 'Others'),
                        **content_dict,
                        "confidence": "High",
                        "found_in_db": True,
                        "used_web_search": False,
                    }
                
                
            # LOW confidence or no matches - use web search
            web_result = await self._web_search(
                query="",
                visual_features=visual_analysis,
                gps_location=gps_location
            )
            
            return {
                "success": True,
                **web_result,
                "confidence": "Low",
                "found_in_db": False,
                "used_web_search": True,
            }
            
        except Exception as e:
            logger.error(f"Identification error: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
            }
    
    
# Singleton instance
_agent_service = None

def get_agent_service() -> LocationIdentificationAgent:
    """Get or create singleton"""
    global _agent_service
    if _agent_service is None:
        _agent_service = LocationIdentificationAgent()
    return _agent_service