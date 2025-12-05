from app.config.settings import settings
from langchain.agents import create_react_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from app.services.pineconeService import get_pinecone_service
from app.config.settings import settings
import google.generativeai as genai
from PIL import Image
import logging, re, json, asyncio
from typing import Dict, Optional, Tuple
from geopy.geocoders import Nominatim
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class LocationIdentificationAgent:
    
    def __init__(self):
        
        self.pinecone_service = get_pinecone_service()
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.vision_model = genai.GenerativeModel(settings.VISION_MODEL)
        self.text_model = genai.GenerativeModel(settings.GEMINI_MODEL)

        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.2
        )

        # Store current context
        self._current_image: Optional[Image.Image] = None
        self._current_gps: Optional[Dict[str, float]] = None
        self._current_embedding: Optional[list] = None
        self._visual_analysis: Optional[str] = None
        
        # Thread pool for CPU-bound operations
        self._executor = ThreadPoolExecutor(max_workers=3)
        
        # Create tools
        self.tools = self._create_tools()
        
        # Create agent
        self.agent_executor = self._create_agent()
    
    def _optimize_image(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        """
        Optimize image for faster processing:
        - Resize to max dimension while maintaining aspect ratio
        - Convert to RGB
        """
        try:
            # Get current dimensions
            width, height = image.size
            
            # Calculate new dimensions if needed
            if max(width, height) > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                
                # Resize with high-quality resampling
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Image resized from {width}x{height} to {new_width}x{new_height}")
            
            # Convert to RGB (only once)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.error(f"Error optimizing image: {str(e)}")
            raise
    
    async def set_image_and_gps(self, image: Image.Image, gps_location: Dict[str, float]):
        """
        Store current image, GPS, and generate embedding in parallel.
        Optimized with image preprocessing and parallel execution.
        """
        try:
            # Validate inputs
            if image is None:
                raise ValueError("Image cannot be None")
            
            if not gps_location or 'lat' not in gps_location or 'lng' not in gps_location:
                gps_location = {'lat': 0.0, 'lng': 0.0}

            self._current_gps = gps_location

            if not isinstance(image, Image.Image):
                raise ValueError(f"Expected PIL.Image, got {type(image)}")
            
            # Optimize image ONCE (resize + RGB conversion)
            loop = asyncio.get_event_loop()
            self._current_image = await loop.run_in_executor(
                self._executor, 
                self._optimize_image, 
                image
            )
            
            # Validate optimized image
            if self._current_image.size[0] == 0 or self._current_image.size[1] == 0:
                raise ValueError(f"Invalid image dimensions: {self._current_image.size}")
            
            # Generate embedding asynchronously
            self._current_embedding = await loop.run_in_executor(
                self._executor,
                self.pinecone_service.generate_image_embedding,
                self._current_image
            )
            
            logger.info(f"Image prepared: {self._current_image.size}, Embedding: {len(self._current_embedding)} dims")
            
        except Exception as e:
            logger.error(f"Failed to set image and GPS: {str(e)}", exc_info=True)
            self._current_image = None
            self._current_embedding = None
            raise ValueError(f"Image preparation failed: {str(e)}")
    
    async def _analyze_image_parallel(self) -> str:
        """Analyze image using Gemini Vision (async wrapper)"""
        try:
            if self._current_image is None:
                return "Error: No image available"
            
            prompt = f"""
                Analyze this image captured in Sri Lanka at GPS: {self._current_gps.get('lat', 'unknown')}, {self._current_gps.get('lng', 'unknown')}.

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
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self._executor,
                lambda: self.text_model.generate_content([prompt, self._current_image])
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Visual analysis error: {str(e)}")
            return f"Error: {str(e)}"
    
    async def _search_database_parallel(self) -> Tuple[list, str]:
        """Search database using embeddings (async wrapper)"""
        try:
            if self._current_embedding is None:
                return [], "Error: No embedding available"
            
            # Prepare filter
            filter_dict = {}
            if self._current_gps:
                lat, lng = self._current_gps.get('lat', 0), self._current_gps.get('lng', 0)
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
                    vector=self._current_embedding,
                    top_k=3,  # Reduced from 5
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
    
    def _generate_content_with_gemini(
        self,
        destination_name: str,
        district_name: str,
        category_name: str,
        db_description: str,
        visual_analysis: str,
        gps_location: dict,
        confidence: str
    ) -> dict:
        """Generate rich content using Gemini for known locations"""
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
            
            response = self.text_model.generate_content(prompt)
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
            return self._get_fallback_content(destination_name, district_name)
        except Exception as e:
            logger.error(f"Content generation error: {str(e)}")
            return self._get_fallback_content(destination_name, district_name)
    
    def _get_fallback_content(self, dest_name: str, district: str) -> dict:
        """Fallback content structure"""
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
    
    def _create_tools(self):
        """Create agent tools"""
        
        tools = [
            Tool(
                name="AnalyzeImageVisually",
                func=self._analyze_image_tool,
                description="""
                Analyzes captured image for visual features, landmarks, architecture.
                Input: "analyze"
                Returns: Detailed visual analysis.
                **USE FIRST** in every workflow.
                """
            ),
            Tool(
                name="SearchDatabaseByImage",
                func=self._search_database_tool,
                description="""
                Searches Sri Lankan destinations using image similarity.
                Filters within 10km if GPS available.
                Input: "search"
                Returns: Top 3 matches with confidence scores.
                **USE AFTER visual analysis**.
                """
            ),
            Tool(
                name="GetNearbyLocations",
                func=self._get_nearby_tool,
                description="""
                Retrieves destinations near GPS coordinates.
                Input: "nearby"
                Returns: Nearby destinations with distances.
                **USE ONLY when database confidence is MEDIUM/LOW**.
                """
            ),
            Tool(
                name="SearchWebForLocation",
                func=self._web_search_tool_wrapper,
                description="""
                Identifies unknown locations via web search + reverse geocoding.
                
                USE ONLY WHEN:
                - Database confidence LOW (<0.75)
                - No database matches
                
                Input: JSON with query/visual_features/gps_location
                Returns: Complete tourist content
                
                FALLBACK TOOL - Don't use if HIGH confidence.
                """
            )
        ]
        
        return tools
    
    def _create_agent(self):
        
        template = """You are an expert Sri Lankan location identification assistant.

            STRICT WORKFLOW:
            1. ALWAYS call AnalyzeImageVisually first
            2. ALWAYS call SearchDatabaseByImage second
            3. Check confidence:
            - HIGH (>0.85): Accept result, skip other tools
            - MEDIUM (0.75-0.85): Call GetNearbyLocations to verify
            - LOW (<0.75): Call GetNearbyLocations then SearchWebForLocation
            4. Provide final answer

            TOOLS: {tools}
            TOOL NAMES: {tool_names}

            Format:
            Question: {input}
            Thought: What to do?
            Action: [tool name]
            Action Input: [input]
            Observation: [output]
            ... (repeat)
            Thought: I have enough info
            Final Answer: [summary]

            Question: {input}
            {agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)
        agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,  # Reduced from 6
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    
    def _analyze_image_tool(self, input_str: str) -> str:

        try:
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(self._analyze_image_parallel())
            self._visual_analysis = result
            return result
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _search_database_tool(self, input_str: str) -> str:
        """Tool wrapper for database search"""
        try:
            loop = asyncio.get_event_loop()
            matches, output = loop.run_until_complete(self._search_database_parallel())
            return output
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _get_nearby_tool(self, input_str: str) -> str:
        """Find nearby destinations using GPS"""
        try:
            if not self._current_gps or 'lat' not in self._current_gps:
                return "Error: GPS not available"

            current_lat = self._current_gps['lat']
            current_lng = self._current_gps['lng']
            radius_km = 5.0

            lat_range = radius_km / 111.0
            lng_range = radius_km / (111.0 * abs(current_lat)) if current_lat != 0 else radius_km / 111.0

            filter_dict = {
                "$and": [
                    {"latitude": {"$gte": current_lat - lat_range, "$lte": current_lat + lat_range}},
                    {"longitude": {"$gte": current_lng - lng_range, "$lte": current_lng + lng_range}},
                ]
            }

            zero_vector = [0.0] * 512
            results = self.pinecone_service.index.query(
                vector=zero_vector,
                top_k=20,
                include_metadata=True,
                namespace='destinationImages',
                filter=filter_dict
            )

            matches = results.get('matches', [])
            if not matches:
                return f"No destinations within {radius_km}km"

            from math import radians, sin, cos, sqrt, atan2

            output = f"Nearby destinations within {radius_km}km:\n\n"
            for i, match in enumerate(matches, 1):
                meta = match['metadata']
                lat2, lon2 = meta.get('latitude', 0), meta.get('longitude', 0)

                lat1r, lon1r = radians(current_lat), radians(current_lng)
                lat2r, lon2r = radians(lat2), radians(lon2)
                dlat, dlon = lat2r - lat1r, lon2r - lon1r
                a = sin(dlat/2)**2 + cos(lat1r)*cos(lat2r)*sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                distance = 6371 * c

                output += f"{i}. {meta.get('name', 'Unknown')} (~{distance:.2f}km)\n"

            return output

        except Exception as e:
            logger.error(f"Nearby search error: {str(e)}")
            return f"Error: {str(e)}"
    
    def _web_search_tool(self, query: str, visual_features: str = None, gps_location: dict = None) -> dict:
        """
        Identify location using reverse geocoding + Gemini with robust error handling
        """
        try:
            lat = gps_location.get("lat") or gps_location.get("latitude") if gps_location else None
            lng = gps_location.get("lng") or gps_location.get("longitude") if gps_location else None
            
            district_name, city_name, village_name, nearby_feature = None, None, None, None
            geocoded_landmark_name = None
            full_address = {}  # NEW: Store full address for debugging

            if lat and lng:
                try:
                    geolocator = Nominatim(user_agent="ceylango_agent")
                    location = geolocator.reverse((lat, lng), language="en", timeout=10)
                    
                    if location and location.raw.get("address"):
                        address = location.raw["address"]
                        full_address = address  # Store for logging
                        
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
                        
                        # Try to get landmark name
                        geocoded_landmark_name = (
                            address.get("amenity") or
                            address.get("tourism") or
                            address.get("attraction") or
                            address.get("historic") or
                            address.get("building") or
                            address.get("monument")
                        )
                        
                        logger.info(f"Geocoded landmark name: {geocoded_landmark_name} address: {address}")
                        
                except Exception as geo_error:
                    logger.warning(f"Geocoding failed: {geo_error}")

            # Build context from available location data
            location_context = ", ".join(filter(None, [
                village_name,
                city_name, 
                district_name
            ])) or "Unknown region in Sri Lanka"

            # ✅ FIX 1: Use geocoded name if available
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
                # ✅ FIX 2: Fallback with better prompt
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
                response = self.text_model.generate_content(prompt)
                
                if not response or not response.text:
                    raise ValueError("Gemini returned empty response")
                
                generated_text = response.text.strip()
                logger.info(f"Gemini raw response (first 200 chars): {generated_text[:200]}")
                
                # Remove markdown fences
                generated_text = re.sub(r'^```json\s*|\s*```$', '', generated_text, flags=re.MULTILINE).strip()
                
                # ✅ FIX 4: Validate JSON before parsing
                if not generated_text:
                    raise ValueError("Generated text is empty after cleanup")
                
                if not generated_text.startswith('{'):
                    logger.warning(f"Response doesn't start with '{{': {generated_text[:100]}")
                    # Try to extract JSON from text
                    json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
                    if json_match:
                        generated_text = json_match.group(0)
                    else:
                        raise ValueError("No JSON object found in response")
                
                content_dict = json.loads(generated_text)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {str(e)}")
                logger.error(f"Generated text: {generated_text}")
                
                # ✅ FIX 5: Fallback to generic content
                return self._create_fallback_content(
                    geocoded_landmark_name or f"Location in {village_name or district_name or 'Sri Lanka'}",
                    district_name,
                    village_name,
                    location_context
                )
            
            except Exception as gemini_error:
                logger.error(f"Gemini generation failed: {str(gemini_error)}")
                return self._create_fallback_content(
                    geocoded_landmark_name or f"Location in {village_name or district_name or 'Sri Lanka'}",
                    district_name,
                    village_name,
                    location_context
                )
            
            # ✅ Return successful result
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
        ✅ NEW: Create generic fallback content when Gemini fails
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



    def _web_search_tool_wrapper(self, input_str: str) -> str:
        """Agent wrapper for web search"""
        try:
            if input_str.strip().startswith('{'):
                params = json.loads(input_str)
                query = params.get("query", "")
                visual_features = params.get("visual_features")
                gps_location = params.get("gps_location")
            else:
                query = input_str
                visual_features = None
                gps_location = None
            
            if not gps_location and self._current_gps:
                gps_location = self._current_gps
            
            if not visual_features:
                visual_features = getattr(self, '_visual_analysis', None)
            
            result_dict = self._web_search_tool(
                query=query,
                visual_features=visual_features,
                gps_location=gps_location
            )
            
            formatted = f"""
                LOCATION IDENTIFIED VIA WEB SEARCH:

                Destination: {result_dict['destination_name']}
                District: {result_dict['district_name']}
                Category: {result_dict['category']}

                Historical Background:
                {result_dict['historical_background']}

                Cultural Significance:
                {result_dict['cultural_significance']}

                What Makes It Special:
                {result_dict['what_makes_it_special']}

                Visitor Experience:
                {result_dict['visitor_experience']}

                Interesting Facts:
                {chr(10).join(f"  • {fact}" for fact in result_dict['interesting_facts'])}

                ---
                STATUS: Web search + geocoding
                CONFIDENCE: Medium
                """
            
            return formatted
            
        except Exception as e:
            logger.error(f"Web search wrapper error: {str(e)}", exc_info=True)
            return f"Error: {str(e)}"
    
    async def identify_and_generate_content(
        self,
        image: Image.Image,
        gps_location: Dict[str, float],
    ) -> Dict:
        """
        Main entry point - orchestrates workflow with parallel processing
        """
        try:
            # Step 1: Prepare image and embedding (async)
            await self.set_image_and_gps(image, gps_location)
            
            # Step 2: Run visual analysis and database search IN PARALLEL
            visual_task = asyncio.create_task(self._analyze_image_parallel())
            db_task = asyncio.create_task(self._search_database_parallel())
            
            visual_analysis, (db_matches, db_output) = await asyncio.gather(visual_task, db_task)
            
            self._visual_analysis = visual_analysis
            
            # Step 3: Determine confidence and decide next steps
            if db_matches:
                top_match = db_matches[0]
                confidence_score = top_match['score']
                
                if confidence_score > 0.85:
                    # HIGH confidence - use database result
                    meta = top_match['metadata']
                    
                    content_dict = self._generate_content_with_gemini(
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
                
                elif confidence_score > 0.75:
                    # MEDIUM confidence - verify with nearby check
                    nearby_output = self._get_nearby_tool("nearby")
                    
                    meta = top_match['metadata']
                    content_dict = self._generate_content_with_gemini(
                        destination_name=meta.get('name', 'Unknown'),
                        district_name=meta.get('district', 'Unknown'),
                        category_name=meta.get('category', 'Others'),
                        db_description=meta.get('description', ''),
                        visual_analysis=visual_analysis,
                        gps_location=gps_location,
                        confidence="Medium"
                    )
                    
                    return {
                        "success": True,
                        "destination_name": meta.get('name', 'Unknown'),
                        "district_name": meta.get('district', 'Unknown'),
                        "category": meta.get('category', 'Others'),
                        **content_dict,
                        "confidence": "Medium",
                        "found_in_db": True,
                        "used_web_search": False,
                    }
            
            # LOW confidence or no matches - use web search
            web_result = self._web_search_tool(
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
    
    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract single-line field"""
        try:
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith(field_name):
                    return line.split(':', 1)[1].strip()
            return ""
        except:
            return ""
    
    def _extract_destination_name(self, db_output: str) -> str:
        """Extract destination from database output"""
        try:
            lines = db_output.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('1.'):
                    name = line[3:].split('[')[0].strip()
                    return name
            return "Unknown"
        except:
            return "Unknown"


# Singleton instance
_agent_service = None

def get_agent_service() -> LocationIdentificationAgent:
    """Get or create singleton"""
    global _agent_service
    if _agent_service is None:
        _agent_service = LocationIdentificationAgent()
    return _agent_service