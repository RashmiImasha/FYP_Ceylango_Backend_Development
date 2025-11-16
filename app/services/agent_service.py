from app.config.settings import settings
from langchain.agents import create_structured_chat_agent
from langchain.agents.agent import AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.services.pineconeService import get_pinecone_service
from app.config.settings import settings
import google.generativeai as genai
from PIL import Image
import logging, re
from typing import Dict, Optional
from geopy.geocoders import Nominatim

logger = logging.getLogger(__name__)

class LocationIdentificationAgent:
    """
    Intelligent agent for identifying locations from images using
    multiple tools: Pinecone search, Vision analysis, GPS verification
    """
    def __init__(self):
        
        self.pinecone_service = get_pinecone_service()

        # initialize vision model
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.vision_model = genai.GenerativeModel(settings.VISION_MODEL)
        self.text_model = genai.GenerativeModel(settings.GEMINI_MODEL)

        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.2  # Lower temperature for more consistent reasoning
        )

        # Store current context
        self._current_image: Optional[Image.Image] = None
        self._current_gps: Optional[Dict[str, float]] = None
        self._current_embedding: Optional[list] = None
        
        # Create tools
        self.tools = self._create_tools()
        
        # Create agent
        self.agent_executor = self._create_agent()
        
        logger.info("LocationIdentificationAgent initialized")
    
    def set_image_and_gps(self, image: Image.Image, gps_location: Dict[str, float]):
        """
        Store the current image, GPS, and generate embedding.
        Must be called before using tools.
        This version is robust against corrupted or invalid images.
        """
        # logger.info(f"Input image type: {type(image)}, is None: {image is None}")
        
        # Validate image input
        if image is None:
            logger.error("set_image_and_gps called with None image")
            raise ValueError("Image cannot be None")
        
        # Ensure we have a valid GPS dictionary
        if not gps_location or 'lat' not in gps_location or 'lng' not in gps_location:
            gps_location = {'lat': 0.0, 'lng': 0.0}

        self._current_gps = gps_location

        # Ensure image is a proper PIL.Image object
        try:
            if not isinstance(image, Image.Image):
                logger.error(f"Input image is not a PIL.Image. Type={type(image)}")
                raise ValueError(f"Expected PIL.Image, got {type(image)}")
            
            # logger.info(f"Before copy: image size={image.size}, mode={image.mode}")
            
            # Create a fresh copy - THIS IS CRITICAL
            self._current_image = image.copy().convert('RGB')
            
            # logger.info(f"After copy: _current_image is None: {self._current_image is None}")
            # logger.info(f"After copy: _current_image size={self._current_image.size}, mode={self._current_image.mode}")
            
            # Validate image dimensions
            if self._current_image.size[0] == 0 or self._current_image.size[1] == 0:
                raise ValueError(f"Invalid image dimensions: {self._current_image.size}")
                
            # logger.info(f"Image validated successfully: {self._current_image.size}, mode={self._current_image.mode}")
            
        except Exception as e:
            logger.error(f"Failed to process input image: {str(e)}", exc_info=True)
            self._current_image = None
            self._current_embedding = None
            raise ValueError(f"Invalid image provided: {str(e)}")

        # Generate embedding 
        try:
            # logger.info(f"_current_image before embedding: is None={self._current_image is None}, type={type(self._current_image)}")
            
            if self._current_image is None:
                raise ValueError("current_image is None before generating embedding!")
            
            # logger.info(f"Calling generate_image_embedding with image size={self._current_image.size}")
            self._current_embedding = self.pinecone_service.generate_image_embedding(self._current_image)
            
            # logger.info(f"Embedding generated successfully. Length: {len(self._current_embedding) if self._current_embedding else 0}")
        
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}", exc_info=True)
            self._current_embedding = None
            raise ValueError(f"Failed to generate image embedding: {str(e)}")        
    
    def _generate_content_with_gemini(
        self,
        destination_name: str,
        district_name: str,
        category_name: str,
        db_description: str,
        visual_analysis: str,
        gps_location: dict,
        confidence: str
    ) -> str:
        """
        Generate rich, accurate content using only Gemini when location is known
        This is used for HIGH/MEDIUM confidence matches from the database
        """
        try:
            logger.info(f"Generating content for known location: {destination_name}")
            
            prompt = f"""You are an expert Sri Lankan tourism guide with deep knowledge of the country's history, culture, and attractions.

            A tourist has captured an image of **{destination_name}** in Sri Lanka.

            CONFIRMED INFORMATION:
            - Location: {destination_name}
            - District: {district_name}
            - Category: {category_name}
            - Confidence: {confidence}
            - GPS Coordinates: {gps_location.get('lat')}, {gps_location.get('lng')}

            DATABASE INFORMATION:
            {db_description if db_description else 'Limited information available in database'}

            VISUAL ANALYSIS:
            {visual_analysis[:600]}

            YOUR TASK:
            Generate an engaging, comprehensive description for tourists visiting {destination_name}.

            REQUIREMENTS:
            1. **Historical Background**: Provide rich historical context about this specific location
            - When was it built/established?
            - Who built it or what civilization/era does it belong to?
            - Historical events that occurred here

            2. **Cultural Significance**: Why is this place important?
            - Religious/cultural importance
            - Role in Sri Lankan heritage
            - UNESCO status (if applicable)

            3. **What Makes It Special**: Unique features and highlights
            - Architectural details
            - Natural beauty or special characteristics
            - Famous elements or attractions within the location

            4. **Visitor Experience**: Practical information tourists need
            - Best time to visit
            - What to see/do there
            - Approximate visit duration
            - Any special tips or recommendations

            5. **Interesting Facts**: 2-3 fascinating facts that most tourists don't know

            GUIDELINES:
            - Use your extensive knowledge about {destination_name} specifically
            - Be accurate and factual (you know Sri Lankan attractions well)
            - Write in an enthusiastic, warm, conversational tone
            - Make tourists excited to visit
            - Length: 250-400 words
            - Use the database info as a foundation, but greatly expand on it with your knowledge

            Generate the description now:
            """
            
            response = self.text_model.generate_content(prompt)
            generated_content = response.text
            
            logger.info(f"Generated {len(generated_content)} chars of content for {destination_name}")
            return generated_content
            
        except Exception as e:
            logger.error(f"Error generating content with Gemini: {str(e)}")
            
            # Fallback to database description if Gemini fails
            if db_description and len(db_description) > 50:
                logger.info("Gemini failed, using database description")
                return db_description
            else:
                return f"{destination_name} is a notable {category_name} in {district_name}, Sri Lanka. This location offers visitors a unique glimpse into Sri Lankan culture and heritage."
        
    def _create_tools(self):
        """Create tools for the agent"""
        
        tools = [
            Tool(
                name="AnalyzeImageVisually",
                func=self._analyze_image_tool,
                description="""
                Analyzes the captured image to identify visual features, landmarks, and architectural elements.
                This tool uses AI vision to describe what's visible in the image.
                Input: "analyze" (no parameters needed, uses current image context)
                Returns: Detailed visual analysis including landmarks, architecture, natural features, and cultural elements.
                USE THIS FIRST to understand what the image shows before searching the database.
                """
            ),
            Tool(
                name="SearchDatabaseByImage",
                func=self._search_database_tool,
                description="""
                Searches the Sri Lankan destinations database using visual similarity matching.
                Compares the captured image with stored location images using CLIP embeddings.
                If GPS location is available, filters results within 10km radius.
                Input: "search" (no parameters needed, uses current image and GPS context)
                Returns: Top 5 matching locations with similarity scores and their descriptions.
                Use this to find if the location exists in the database.
                """
            ),
            Tool(
                name="GetNearbyLocations",
                func=self._get_nearby_tool,
                description="""
                Retrieves known destinations near the current GPS coordinates from the database.
                This helps identify what landmarks or attractions are in the vicinity.
                Input: "nearby" (no parameters needed, uses current GPS context)
                Returns: List of nearby destinations with names, descriptions, and distances.
                Use this when image search has low confidence but GPS is available.
                """
            ),
            Tool(
                name="SearchWebForLocation",
                func=self._web_search_tool,
                description="""
                Searches the internet for information about a location.
                Use this when database search confidence is low or when you need additional context.
                Input: Location name or description (e.g., "temple in Kandy district")
                Returns: Historical and cultural information from web sources.
                Use as a fallback when database confidence < 0.75 or for additional verification.
                """
            )
        ]
        
        return tools

    def _create_agent(self):
        """Create the reasoning agent"""
        
        system_template = """You are an expert Sri Lankan tourism AI assistant specializing in location identification from images.
        Your mission: Accurately identify locations from photos captured by tourists and generate engaging historical/cultural content.
        WORKFLOW (Follow strictly):

        1. **Visual Analysis First**
        - Always start with AnalyzeImageVisually to understand what's in the photo
        - Identify key visual features: temples, beaches, mountains, forts, etc.

        2. **Database Search**
        - Use SearchDatabaseByImage to find matches in the verified database
        - Evaluate confidence scores:
            * Score > 0.85: HIGH confidence - Trust database information
            * Score 0.75-0.85: MEDIUM confidence - Verify with nearby locations
            * Score < 0.75: LOW confidence - Use web search or nearby locations

        3. **Decision Logic**
        HIGH confidence (>0.85):
        - Accept database match
        - Use stored description and history
        - Mention the matched location name
        
        MEDIUM confidence (0.75-0.85):
        - Check GetNearbyLocations for verification
        - If nearby location names match visual features, proceed
        - Otherwise, use SearchWebForLocation
        
        LOW confidence (<0.75):
        - Use GetNearbyLocations to understand the area
        - If visual features match nearby landmarks, identify
        - Use SearchWebForLocation for additional context
        - If still uncertain, state "Unknown" clearly

        4. **Content Generation**
        - Create engaging, tourist-friendly content in the requested language
        - Include: Historical significance, cultural importance, interesting facts
        - Keep tone informative yet conversational
        - Length: 150-300 words

        CRITICAL RULES:
        - NEVER fabricate location names or history
        - If uncertain, clearly state uncertainty level
        - Always cross-reference visual analysis with database/nearby results
        - For famous Sri Lankan landmarks (Sigiriya, Temple of Tooth, Galle Fort), be confident with high-score matches
        - Consider GPS accuracy limitations (±50-100m typical)

        Current context:
        - Date: September 30, 2025
        - User is in Sri Lanka
        - GPS accuracy: ±50m typical for mobile devices
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template + """
                TOOLS AVAILABLE:
                {tools}

                TOOL NAMES:
                {tool_names}

                Always choose the best tool based on context. """),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_structured_chat_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    
    def _analyze_image_tool(self, input_str: str) -> str:
        """Tool: Analyze image using Gemini Vision"""
        try:
            if self._current_image is None:
                return "Error: No image available for analysis"
            
            image_copy = self._current_image.copy()
            
            prompt = f"""
            Analyze this image captured in Sri Lanka at GPS coordinates: {self._current_gps.get('lat', 'unknown')}, {self._current_gps.get('lng', 'unknown')}.
            
            Provide a structured analysis:
            
            1. **Main Subject**: What is the primary landmark or location? Be specific.
            2. **Architecture/Structure**: Describe buildings, monuments, or structures (e.g., Buddhist stupa, colonial building, modern structure)
            3. **Natural Features**: Landscape elements (mountains, water, vegetation, beaches)
            4. **Cultural/Religious Elements**: Any visible symbols, statues, inscriptions, religious markers
            5. **Distinctive Features**: Unique identifiers that could help pinpoint this exact location
            6. **Time Period**: If historical, estimate the era (ancient, colonial, modern)
            7. **Location Category**: Best category (Temple, Beach, Mountain, Fort, National Park, City, Village, etc.)
            8. **Visible Text**: Any signs, boards, or inscriptions visible (crucial for identification)
            
            Focus on factual observations. For Sri Lankan context, consider famous landmarks like:
            - Ancient cities: Sigiriya, Polonnaruwa, Anuradhapura
            - Religious sites: Temple of Tooth Kandy, Dambulla Cave Temple
            - Colonial: Galle Fort, Colombo Fort
            - Natural: Ella, Nuwara Eliya, Yala, Horton Plains
            
            Be extremely detailed and precise.
            """            
            response = self.text_model.generate_content([prompt, image_copy])            
            return response.text
        
        except Exception as e:
            logger.error(f"Error in visual analysis: {str(e)}")
            return f"Error analyzing image: {str(e)}"
    
    def _search_database_tool(self, input_str: str) -> str:
        """Tool: Search Pinecone database using image embeddings"""
        
        if self._current_embedding is None:
            return "Error: No image embedding available for search"
        
        try:                
            # Apply GPS filter if available
            filter_dict = {}
            if self._current_gps:
                lat, lng = self._current_gps.get('lat', 0), self._current_gps.get('lng', 0)
                radius_km = 10.0
                lat_range = radius_km / 111.0
                lng_range = radius_km / (111.0 * abs(lat)) if lat != 0 else radius_km / 111.0
                filter_dict = {
                    "$and": [
                        {"latitude": {"$gte": lat - lat_range}},
                        {"latitude": {"$lte": lat + lat_range}},
                        {"longitude": {"$gte": lng - lng_range}},
                        {"longitude": {"$lte": lng + lng_range}},
                    ]
                }

            # Query Pinecone for similar images
            results = self.pinecone_service.index.query(
                vector=self._current_embedding,
                top_k=5,
                include_metadata=True,
                namespace='destinationImages',
                filter=filter_dict
            )

            matches = results.get("matches", [])
            if not matches:
                return "No matching locations found in database within 10km radius."

            # Format output for agent
            output = f"Found {len(matches)} potential matches:\n\n"
            for i, match in enumerate(matches, 1):
                score = match['score']
                meta = match['metadata']

                confidence_level = "HIGH" if score > 0.85 else "MEDIUM" if score > 0.75 else "LOW"

                output += f"{i}. {meta.get('name', 'Unknown')} [{confidence_level} CONFIDENCE: {score:.2%}]\n"
                output += f"   District: {meta.get('district', 'Unknown')}\n"
                output += f"   Category: {meta.get('category', 'Unknown')}\n"
                output += f"   Description: {meta.get('description', 'No description')}\n"
                output += f"   GPS: {meta.get('latitude', 0):.4f}, {meta.get('longitude', 0):.4f}\n\n"

            return output

        except Exception as e:
            logger.error(f"Error in database search: {str(e)}")
            return f"Error searching database: {str(e)}"

    def _get_nearby_tool(self, input_str: str) -> str:
        """Tool: Find nearby destinations using current GPS """
        try:
            if not self._current_gps or 'lat' not in self._current_gps:
                return "Error: GPS location not available."

            current_lat = self._current_gps['lat']
            current_lng = self._current_gps['lng']
            radius_km = 5.0  # Search radius

            # Query Pinecone purely using metadata filters (no embedding)
            lat_range = radius_km / 111.0  # Roughly 1° latitude ≈ 111km
            lng_range = radius_km / (111.0 * abs(current_lat)) if current_lat != 0 else radius_km / 111.0

            filter_dict = {
                "$and": [
                    {"latitude": {"$gte": current_lat - lat_range}},
                    {"latitude": {"$lte": current_lat + lat_range}},
                    {"longitude": {"$gte": current_lng - lng_range}},
                    {"longitude": {"$lte": current_lng + lng_range}},
                ]
            }

            # Do a dummy query with a zero vector (since we’re not using embeddings)
            zero_vector = [0.0] * 512  # Match your embedding dimension
            results = self.pinecone_service.index.query(
                vector=zero_vector,
                top_k=20,
                include_metadata=True,
                namespace='destinationImages',
                filter=filter_dict
            )

            matches = results.get('matches', [])
            if not matches:
                return "No destinations found within 5km of your current location."

            # Format results
            output = f"Nearby destinations within {radius_km} km:\n\n"

            from math import radians, sin, cos, sqrt, atan2

            for i, match in enumerate(matches, 1):
                meta = match['metadata']
                lat2 = meta.get('latitude', 0)
                lon2 = meta.get('longitude', 0)

                # Haversine distance calculation
                lat1r, lon1r = radians(current_lat), radians(current_lng)
                lat2r, lon2r = radians(lat2), radians(lon2)
                dlat, dlon = lat2r - lat1r, lon2r - lon1r
                a = sin(dlat / 2)**2 + cos(lat1r) * cos(lat2r) * sin(dlon / 2)**2
                c = 2 * atan2(sqrt(a), sqrt(1 - a))
                distance = 6371 * c  # km

                output += f"{i}. {meta.get('name', 'Unknown')} (~{distance:.2f} km away)\n"
                output += f"   Latitude: {lat2}, Longitude: {lon2}\n\n"

            return output

        except Exception as e:
            logger.error(f"Error getting nearby locations: {str(e)}")
            return f"Error retrieving nearby locations: {str(e)}"

    def _web_search_tool(self, query: str, visual_features: str = None, gps_location: dict = None) -> dict:
        """
        Tool: Identify Sri Lankan location and generate detailed tourist description 
        using Gemini reasoning + reverse geocoding context.
        """
        try:
            # --- Step 1: Reverse geocode latitude and longitude ---
            lat = gps_location.get("lat") if gps_location else None
            lng = gps_location.get("lng") if gps_location else None
            district_name, city_name, nearby_feature = None, None, None

            if lat and lng:
                try:
                    geolocator = Nominatim(user_agent="ceylango_agent")
                    location = geolocator.reverse((lat, lng), language="en", timeout=10)
                    if location and location.raw.get("address"):
                        address = location.raw["address"]
                        district_name = (
                            address.get("state_district")
                            or address.get("county")
                            or address.get("state")
                            or address.get("region")
                        )
                        city_name = (
                            address.get("city")
                            or address.get("town")
                            or address.get("village")
                            or address.get("hamlet")
                        )
                        nearby_feature = (
                            address.get("tourism")
                            or address.get("attraction")
                            or address.get("road")
                            or address.get("suburb")
                        )
                except Exception as geo_error:
                    logger.warning(f"Reverse geocoding failed: {geo_error}")

            # Combine available hints
            geo_context = ", ".join(filter(None, [nearby_feature, city_name, district_name])) or "Unknown region in Sri Lanka"
            logger.info(f"Reverse geocoded context: {geo_context}")

            prompt = f"""
                You are a Sri Lankan tourism assistant AI.

                Use the following context to reason about the most likely location in the image
                and generate a detailed tourist description.

                CONTEXT INFORMATION:
                - Latitude: {lat or 'unknown'}
                - Longitude: {lng or 'unknown'}
                - Reverse-geocoded region: {geo_context}
                - District: {district_name or 'unknown'}
                - Query hint: "{query if query else 'None'}"
                - Visual analysis: {visual_features if visual_features else 'No visual data.'}

                GUIDANCE:
                1. Assume the image is from **within or near the {district_name or 'given'} District**.
                2. Give higher confidence to landmarks or attractions that actually exist in that district.
                3. Begin your answer with the line:
                **Location:** <name of the place or landmark>
                4. Then write a 200–350-word tourist-style description including:
                - Historical and cultural significance
                - Key attractions or rituals
                - Interesting facts
                - Visiting tips and etiquette
                5. If uncertain of the exact name, describe the most probable landmark or area in {district_name or 'the identified district'}.
                6. Avoid referencing places from other districts unless clearly justified by GPS proximity.
                """

            # --- Step 3: Generate reasoning output using Gemini ---
            response = self.text_model.generate_content(prompt)
            generated_text = response.text.strip() if response and response.text else ""

            # --- Step 4: Extract destination name from Gemini output ---
            destination_name = "Unknown"
            match = re.search(r"(?:\*\*)?Location:(?:\*\*)?\s*(.+)", generated_text)
            if match:
                destination_name = match.group(1).strip()
                destination_name = re.sub(r"[\.\-\–]+$", "", destination_name)

            # --- Step 5: Handle fallbacks gracefully ---
            if not generated_text:
                generated_text = (
                    f"This area near coordinates ({lat}, {lng}) appears to be around {geo_context}. "
                    f"The visual features suggest {visual_features[:200] if visual_features else 'a scenic location'}. "
                    "It likely represents a site of cultural, historical, or natural importance typical of Sri Lanka."
                )

            if not district_name:
                district_name = "Unknown"
            
            return {
                "destination_name": destination_name or "Unknown",
                "district_name": district_name or "Unknown",
                "description": generated_text,
            }

        except Exception as e:
            logger.error(f"Error in web search tool: {str(e)}", exc_info=True)
            return {
                "destination_name": "Unknown",
                "district_name": "Unknown",
                "description": f"Error generating location content: {str(e)}",
            }
    
    async def identify_and_generate_content(
        self,
        image: Image.Image,
        gps_location: Dict[str, float],
    ) -> Dict:
        """
        Identify location and generate tourist-friendly content from a single image + GPS.
        Simplified: avoids AgentExecutor and agent_scratchpad.
        """
        try:
            self.set_image_and_gps(image, gps_location)
            
            visual_analysis = self._analyze_image_tool("analyze")
            # logger.info(f"Visual analysis result (first 200 chars): {visual_analysis[:200]}")
            
            db_search_output = self._search_database_tool("search")
            # logger.info(f"Database search output:\n{db_search_output}")
            
            nearby_output = self._get_nearby_tool("nearby")
            # logger.info(f"Nearby search output (first 200 chars): {nearby_output[:200]}")
            
            #  Decide best location
            confidence = "Medium"
            destination_name = "Unknown"
            district_name = "Unknown"  
            db_description = ""          
            category_name = "Others"
            found_in_db = False
            use_web_search = False

            logger.info(f"Checking confidence in db_search_output...")
            lines = db_search_output.splitlines()

            if "HIGH CONFIDENCE" in db_search_output:
                logger.info("HIGH CONFIDENCE match found!")
                confidence = "High"
                found_in_db = True               
                
                # for i, line in enumerate(lines):
                #     logger.info(f"Line {i}: {line}")
                #     if "[HIGH CONFIDENCE" in line :
                #         parts = line.split("[")
                #         if len(parts) > 0:
                #             # Remove leading numbers and dots (e.g., "1. ")
                #             destination_name = parts[0].strip()
                #             # Remove leading number pattern like "1. ", "2. ", etc.
                #             # import re
                #             destination_name = re.sub(r'^\d+\.\s*', '', destination_name)
                                        
               
                #     if line.strip().startswith("district:"):
                #         district_name = line.replace("district:", "").strip()

                #     if line.strip().startswith("category:"):
                #         category_name = line.replace("category:", "").strip()
                    
                #     if line.strip().startswith("description:"):
                #         db_description = line.replace("description:", "").strip()
                    
                #     if (destination_name != "Unknown" and 
                #         district_name != "Unknown" and 
                #         category_name != "Unknown"):
                #         # logger.info("All data extracted from HIGH confidence match")
                #         break

                for line in lines:
                    line_clean = line.strip()
                    line_lower = line_clean.lower()

                    if "[high confidence" in line_lower:
                        parts = line_clean.split("[")
                        if len(parts) > 0:
                            destination_name = re.sub(r'^\d+\.\s*', '', parts[0].strip())

                    if line_lower.startswith("district:"):
                        district_name = line_clean.split(":", 1)[1].strip()

                    if line_lower.startswith("category:"):
                        category_name = line_clean.split(":", 1)[1].strip()

                    if line_lower.startswith("description:"):
                        db_description = line_clean.split(":", 1)[1].strip()
                        
            elif "MEDIUM CONFIDENCE" in db_search_output:
                logger.info("MEDIUM CONFIDENCE match found!")
                confidence = "Medium"
                found_in_db = True

                for line in lines:
                    line_clean = line.strip()
                    line_lower = line_clean.lower()

                    if "[high confidence" in line_lower:
                        parts = line_clean.split("[")
                        if len(parts) > 0:
                            destination_name = re.sub(r'^\d+\.\s*', '', parts[0].strip())

                    if line_lower.startswith("district:"):
                        district_name = line_clean.split(":", 1)[1].strip()

                    if line_lower.startswith("category:"):
                        category_name = line_clean.split(":", 1)[1].strip()

                    if line_lower.startswith("description:"):
                        db_description = line_clean.split(":", 1)[1].strip()
            else:
                confidence = "Low"
                use_web_search = True
                found_in_db = False

            logger.info(f"Found in database: {found_in_db}, Use web search: {use_web_search}")
            description = ""

            if found_in_db and destination_name != "Unknown":
                
                description = self._generate_content_with_gemini(
                    destination_name=destination_name,
                    district_name=district_name,
                    category_name=category_name,
                    db_description=db_description,
                    visual_analysis=visual_analysis,
                    gps_location=gps_location,
                    confidence=confidence
                )
                use_web_search = False
            
            else:      
                logger.info("Use web search tool for unknown location...")
                web_result = self._web_search_tool(
                    query=destination_name ,
                    visual_features=visual_analysis,
                    gps_location=gps_location
                )
                use_web_search = True

                destination_name = web_result.get("destination_name", "Unknown")
                district_name = web_result.get("district_name", "Unknown")
                description = web_result.get("description", "")

            result = {
                "success": True,
                "destination_name": destination_name,
                "district_name": district_name,
                "category": category_name,
                "description": description,
                "confidence": confidence,
                "found_in_db": found_in_db,
                "used_web_search": use_web_search,
                "visual_analysis": visual_analysis,
                "database_output": db_search_output,
                "nearby_output": nearby_output
            }
            
            logger.info(f"Final extracted values: name={destination_name}, district={district_name}, category={category_name}, confidence={confidence}")
            # logger.info(f"Result: {result['destination_name']} - {result['confidence']} confidence")
            return result

        except Exception as e:
            logger.error(f"Error in agent execution: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),                                
            }
    

# Singleton instance
_agent_service = None

def get_agent_service() -> LocationIdentificationAgent:
    """Get or create LocationIdentificationAgent singleton"""
    global _agent_service
    if _agent_service is None:
        _agent_service = LocationIdentificationAgent()
    return _agent_service