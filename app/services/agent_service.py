from app.config.settings import settings
from langchain.agents import create_react_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from app.services.pineconeService import get_pinecone_service
from app.config.settings import settings
import google.generativeai as genai
from PIL import Image
import logging, re
from typing import Dict, Optional
from geopy.geocoders import Nominatim

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
        
        # logger.info("LocationIdentificationAgent initialized")
    
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
    ) -> dict:
        """
        Generate rich, accurate content using only Gemini when location is known
        Returns structured dict with separate fields
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
            Generate structured information for tourists visiting {destination_name}.

            You MUST respond ONLY with a valid JSON object in this exact format (no markdown, no preamble):

            {{
            "historical_background": "when it was built/established, who built it, historical events that occurred here",
            "cultural_significance": "religious/cultural importance, role in Sri Lankan heritage, UNESCO status if applicable",
            "what_makes_it_special": "unique features, architectural details, natural beauty, famous elements",
            "visitor_experience": "best time to visit, what to see/do, visit duration, special tips",
            "interesting_facts": [
                    "fascinating fact 1",
                    "fascinating fact 2",
                    "fascinating fact 3"
                ]            
            }}

            GUIDELINES:
            - Use your extensive knowledge about {destination_name} specifically
            - Be accurate and factual
            - Write in an enthusiastic, warm, conversational tone
            - Each text field should be well-crafted sentences
            - Provide exactly 3 interesting facts
            - Return ONLY the JSON object, nothing else

            Generate the JSON now:
            """
            
            response = self.text_model.generate_content(prompt)
            generated_text = response.text.strip()
            
            # Remove markdown code fences if present
            generated_text = re.sub(r'^```json\s*|\s*```$', '', generated_text, flags=re.MULTILINE).strip()
            
            # Parse JSON
            import json
            content_dict = json.loads(generated_text)

            for field in ['historical_background', 'cultural_significance', 'what_makes_it_special', 'visitor_experience']:
                value = content_dict.get(field, "")
                if isinstance(value, list):
                    # Convert list to paragraph
                    content_dict[field] = " ".join(value)
                    logger.warning(f"Converted {field} from list to string for {destination_name}")
            
            #  VALIDATION: Ensure interesting_facts is a list
            facts = content_dict.get('interesting_facts', [])
            if not isinstance(facts, list):
                content_dict['interesting_facts'] = [str(facts)]
            
            logger.info(f"generate_content_with_gemini: Generated structured content for {destination_name}")
            return content_dict
            
        except json.JSONDecodeError as e:
            logger.error(f"generate_content_with_gemini: JSON parsing error: {str(e)}")
            
            return {
                "historical_background": f"{destination_name} is a notable in {district_name}, Sri Lanka.",
                "cultural_significance": "This location offers visitors a unique glimpse into Sri Lankan culture and heritage.",
                "what_makes_it_special": "A remarkable destination worth visiting.",
                "visitor_experience": "Visit during daylight hours for the best experience.",
                "interesting_facts": [
                    "This is one of Sri Lanka's important cultural sites.",
                    "The location attracts both local and international visitors.",
                    "Photography is usually permitted here."
                ]
            }
        
        except Exception as e:
            logger.error(f"generate_content_with_gemini: Error generating content with Gemini: {str(e)}")
            return {
                "historical_background": f"{destination_name} is located in {district_name}.",
                "cultural_significance": "This site holds cultural importance in Sri Lanka.",
                "what_makes_it_special": "A unique destination to explore.",
                "visitor_experience": "Plan your visit accordingly.",
                "interesting_facts": [
                    "More information available on-site.",
                    "Local guides can provide detailed insights.",
                    "Check visiting hours before arrival."
                ]
            }   
         
    def _create_tools(self):
        """Create tools for the agent"""
        
        tools = [
            Tool(
                name="AnalyzeImageVisually",
                func=self._analyze_image_tool,
                description="""
                Analyzes the captured image to identify visual features, landmarks, and architectural elements.
                Input: "analyze" (no parameters needed, uses current image context)
                Returns: Detailed visual analysis including landmarks, architecture, natural features, and cultural elements.
                **USE THIS FIRST** in every workflow.
                """
            ),
            Tool(
                name="SearchDatabaseByImage",
                func=self._search_database_tool,
                description="""
                Searches the Sri Lankan destinations database using image similarity matching.
                Compares the captured image with stored location images using CLIP embeddings.
                If GPS location is available, filters results within 10km radius.
                Input: "search" (no parameters needed, uses current image and GPS context)
                Returns: Top 5 matching locations with similarity scores and their descriptions.
                **USE AFTER visual analysis**.               
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
                **USE WHEN** database confidence is MEDIUM/LOW for verification.                """
            ),
            Tool(
            name="SearchWebForLocation",
            func=self._web_search_tool_wrapper,  
            description="""
                Identifies unknown Sri Lankan locations using web search + reverse geocoding.
            
                ONLY USE WHEN:
                - SearchDatabaseByImage returned LOW confidence (<0.75)
                - No matches found in database
                - Nearby verification failed
                
                Input formats accepted:
                1. Plain text: "temple with golden roof"
                2. JSON: {"query": "temple", "visual_features": "...", "gps_location": {...}}
                
                This tool:
                - Uses reverse geocoding to identify the region
                - Generates complete tourist content (history, culture, facts)
                - Returns structured information ready for final output
                
                FALLBACK TOOL - Do not use if database has HIGH confidence match.
                """
            )
        ]
        
        return tools

    def _create_agent(self):
        """Create ReAct agent for location identification"""
        
        template = """You are an expert Sri Lankan location identification assistant with access to multiple tools.

            Your task: Identify locations from images and generate tourist-friendly content.

            STRICT WORKFLOW:
            1. ALWAYS call AnalyzeImageVisually first
            2. ALWAYS call SearchDatabaseByImage second
            3. Check confidence score:
            - HIGH (>0.85): Accept database result
            - MEDIUM (0.75-0.85): Call GetNearbyLocations to verify
            - LOW (<0.75): Call GetNearbyLocations then SearchWebForLocation
            4. Provide final answer with all required fields

            AVAILABLE TOOLS:
            {tools}

            TOOL NAMES: {tool_names}

            Use this format:

            Question: {input}
            Thought: What should I do first?
            Action: [tool name from {tool_names}]
            Action Input: [input for the tool]
            Observation: [tool output]
            ... (repeat Thought/Action/Observation as needed)
            Thought: I have gathered enough information
            Final Answer: [structured summary of findings]

            Begin!

            Question: {input}
            {agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)

        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=6,
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
                
                CRITICAL RULES:
                - Focus ONLY on what you can SEE in the image
                - DO NOT guess the location name based on GPS coordinates alone
                - DO NOT mention specific temple names unless you can see a sign/text in the image
                - Describe visual features objectively without making location assumptions
                - If you see text/signs in the image, mention them exactly as written
                
                For Sri Lankan context, note if visual features resemble famous landmark types like:
                - Ancient city ruins (Sigiriya-style, Polonnaruwa-style, etc.)
                - Religious architecture (Buddhist stupa, Hindu kovil, colonial church, etc.)
                - Colonial structures (Dutch fort-style, British colonial, etc.)
                - Natural formations (specific mountain formations, beaches, etc.)
                
                Be extremely detailed and precise about VISIBLE features only.
                """ 
            
            response = self.text_model.generate_content([prompt, image_copy])      
            analysis_result = response.text

            self._visual_analysis = analysis_result    
            return analysis_result
        
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
        Tool: Identify Sri Lankan location and generate structured tourist description 
        using Gemini reasoning + reverse geocoding context.
        """
        try:
            # [Keep existing reverse geocoding code...]
            lat = gps_location.get("lat") or gps_location.get("latitude") if gps_location else None
            lng = gps_location.get("lng") or gps_location.get("longitude") if gps_location else None
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

            geo_context = ", ".join(filter(None, [nearby_feature, city_name, district_name])) or "Unknown region in Sri Lanka"
            logger.info(f"Reverse geocoded context: {geo_context}")

            prompt = f"""
            You are a Sri Lankan tourism assistant AI.

            Use the following context to reason about the most likely location in the image
            and generate structured tourist information.

            CONTEXT INFORMATION:
            - Latitude: {lat or 'unknown'}
            - Longitude: {lng or 'unknown'}
            - Reverse-geocoded region: {geo_context}
            - District: {district_name or 'unknown'}
            - Query hint: "{query if query else 'None'}"
            - Visual analysis: {visual_features if visual_features else 'No visual data.'}

            GUIDANCE:
            1. Assume the image is from **within or near the {district_name or 'given'} District**.
            2. Give higher confidence to landmarks that actually exist in that district.

            IMPORTANT:
            - The coordinates map to: {district_name or "unknown district"}.
            - Therefore the landmark MUST be located in THIS district.
            - NEVER select a landmark from Colombo, Kandy, Galle, or any other district if GPS is present.
            - GPS location is the MOST authoritative signal.
            - Always prioritise {district_name or "the reverse-geocoded district"} over visual resemblance and text description.

            You MUST respond ONLY with a valid JSON object in this exact format (no markdown, no preamble):

            {{
            "destination_name": "name of the place or landmark",
            "category": "Temple/Beach/Fort/Mountain/National Park/City/Village/Ancient City/Others",  
            "historical_background": "history, who built it, historical events",
            "cultural_significance": "religious/cultural importance, heritage significance",
            "what_makes_it_special": "unique features, attractions, key elements",
            "visitor_experience": "visiting tips, best time, what to see/do",
            "interesting_facts": [
                "fascinating fact 1",
                "fascinating fact 2",
                "fascinating fact 3"
            ]
            }}

            If uncertain of exact name, provide the most probable landmark in {district_name or 'the district'}.
            Return ONLY the JSON object, nothing else.
            """

            response = self.text_model.generate_content(prompt)
            generated_text = response.text.strip() if response and response.text else ""
            
            # Remove markdown code fences
            generated_text = re.sub(r'^```json\s*|\s*```$', '', generated_text, flags=re.MULTILINE).strip()
            
            # Parse JSON
            import json
            content_dict = json.loads(generated_text)
            
            return {
                "destination_name": content_dict.get("destination_name", "Unknown"),
                "district_name": district_name or content_dict.get("district_name", "Unknown"),
                "category": content_dict.get("category", "Others"), 
                "historical_background": content_dict.get("historical_background", ""),
                "cultural_significance": content_dict.get("cultural_significance", ""),
                "what_makes_it_special": content_dict.get("what_makes_it_special", ""),
                "visitor_experience": content_dict.get("visitor_experience", ""),
                "interesting_facts": content_dict.get("interesting_facts", [])
            }

        except Exception as e:
            logger.error(f"Error in web search tool: {str(e)}", exc_info=True)
            return {
                "destination_name": "Unknown",
                "district_name": "Unknown",
                "category": "Others",
                "historical_background": "",
                "cultural_significance": "",
                "what_makes_it_special": "",
                "visitor_experience": "",
                "interesting_facts": []
            }
    
    def _web_search_tool_wrapper(self, input_str: str) -> str:
        """
        Wrapper for LangChain agent to call the web search tool.
        
        Agent provides string input, this converts it to proper parameters,
        calls the ORIGINAL _web_search_tool (which uses geolocator),
        and returns formatted string output for agent to read.
        """
        try:
            import json
            
            # Parse agent's input
            if input_str.strip().startswith('{'):
                # Agent provided JSON
                params = json.loads(input_str)
                query = params.get("query", "")
                visual_features = params.get("visual_features")
                gps_location = params.get("gps_location")
            else:
                # Agent provided plain text query
                query = input_str
                visual_features = None
                gps_location = None
            
            # Use stored context if not provided
            if not gps_location and self._current_gps:
                gps_location = self._current_gps
            
            # Get visual features from stored context if available
            if not visual_features:
                # Try to find visual analysis from previous tool calls
                # (You might need to store this in self._visual_analysis)
                visual_features = getattr(self, '_visual_analysis', None)
            
            logger.info(f"Web search wrapper called with query: {query}")
            logger.info(f"GPS location: {gps_location}")
            
            # ✅ Call the ORIGINAL method (which uses geolocator!)
            result_dict = self._web_search_tool(
                query=query,
                visual_features=visual_features,
                gps_location=gps_location
            )
            
            # ✅ Convert dict response to formatted string for agent
            formatted_output = f"""
                LOCATION IDENTIFIED VIA WEB SEARCH:

                Destination Name: {result_dict['destination_name']}
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
                STATUS: Content generated successfully using web search + reverse geocoding
                CONFIDENCE: Medium (web-based identification)
                """
            
            logger.info(f"Web search identified: {result_dict['destination_name']}")
            return formatted_output
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in wrapper: {str(e)}")
            return f"Error: Invalid input format. Expected JSON or plain text query. Error: {str(e)}"
        
        except Exception as e:
            logger.error(f"Error in web search wrapper: {str(e)}", exc_info=True)
            return f"Error identifying location via web search: {str(e)}"
        
    async def identify_and_generate_content(
        self,
        image: Image.Image,
        gps_location: Dict[str, float],
    ) -> Dict:
        """
        Let the agent orchestrate the workflow intelligently.
        """
        try:
            # Set context
            self.set_image_and_gps(image, gps_location)
            
            # Prepare agent input
            agent_input = f"""
            Identify the location in this image and generate tourist content.
            
            GPS Coordinates: {gps_location.get('lat')}, {gps_location.get('lng')}
            
            Follow the workflow:
            1. Analyze the image visually
            2. Search the database using image similarity
            3. If confidence is MEDIUM/LOW, check nearby locations
            4. If still uncertain, use web search to identify and generate content
            5. Compile the final structured output
            
            Return structured information suitable for tourists.
            """
            
            # Run agent
            result = await self.agent_executor.ainvoke({
                "input": agent_input
            })
            
            # Parse agent output
            agent_output = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])
            
            # Determine data sources from intermediate steps
            found_in_db = False
            used_web_search = False
            confidence = "Low"
            
            for step in intermediate_steps:
                tool_name = step[0].tool if hasattr(step[0], 'tool') else ""
                tool_output = str(step[1])
                
                if tool_name == "SearchDatabaseByImage":
                    if "HIGH CONFIDENCE" in tool_output:
                        found_in_db = True
                        confidence = "High"
                    elif "MEDIUM CONFIDENCE" in tool_output:
                        found_in_db = True
                        confidence = "Medium"
                
                if tool_name == "SearchWebForLocation":
                    used_web_search = True
            
            # Parse structured content from agent output
            # (You can improve this with structured output parsing)
            parsed_content = self._parse_agent_output(agent_output, intermediate_steps)
            
            return {
                "success": True,
                "destination_name": parsed_content.get("destination_name", "Unknown"),
                "district_name": parsed_content.get("district_name", "Unknown"),
                "category": parsed_content.get("category", "Others"),
                "historical_background": parsed_content.get("historical_background", ""),
                "cultural_significance": parsed_content.get("cultural_significance", ""),
                "what_makes_it_special": parsed_content.get("what_makes_it_special", ""),
                "visitor_experience": parsed_content.get("visitor_experience", ""),
                "interesting_facts": parsed_content.get("interesting_facts", []),
                "confidence": confidence,
                "found_in_db": found_in_db,
                "used_web_search": used_web_search,
            }
            
        except Exception as e:
            logger.error(f"Agent execution error: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
            }
    
    def _parse_agent_output(self, agent_output: str, intermediate_steps: list) -> dict:
        """
        Extract structured content from agent's reasoning and tool outputs.
        """
        parsed = {
            "destination_name": "Unknown",
            "district_name": "Unknown",
            "category": "Others",
            "historical_background": "",
            "cultural_significance": "",
            "what_makes_it_special": "",
            "visitor_experience": "",
            "interesting_facts": []
        }
        
        # Check if web search was used
        web_search_used = False
        for step in intermediate_steps:
            tool_name = step[0].tool if hasattr(step[0], 'tool') else ""
            
            if tool_name == "SearchWebForLocation":
                web_search_used = True
                tool_output = str(step[1])
                
                # FIXED: Match actual field names in wrapper output
                if "Destination Name:" in tool_output:
                    parsed["destination_name"] = self._extract_field(tool_output, "Destination Name:")
                if "District:" in tool_output:
                    parsed["district_name"] = self._extract_field(tool_output, "District:")
                if "Category:" in tool_output:  
                    parsed["category"] = self._extract_field(tool_output, "Category:")
                            
                if "Historical Background:" in tool_output:
                    parsed["historical_background"] = self._extract_multiline_field(tool_output, "Historical Background:", "Cultural Significance:")
                if "Cultural Significance:" in tool_output:
                    parsed["cultural_significance"] = self._extract_multiline_field(tool_output, "Cultural Significance:", "What Makes It Special:")
                if "What Makes It Special:" in tool_output:
                    parsed["what_makes_it_special"] = self._extract_multiline_field(tool_output, "What Makes It Special:", "Visitor Experience:")
                if "Visitor Experience:" in tool_output:
                    parsed["visitor_experience"] = self._extract_multiline_field(tool_output, "Visitor Experience:", "Interesting Facts:")
                if "Interesting Facts:" in tool_output:
                    # FIXED: Extract bullet points correctly
                    parsed["interesting_facts"] = self._extract_bullet_points(tool_output, "Interesting Facts:")
                
                break  # Web search has all info, no need to continue
        
        # If web search wasn't used, extract from database + generate content
        if not web_search_used:
            for step in intermediate_steps:
                tool_name = step[0].tool if hasattr(step[0], 'tool') else ""
                
                if tool_name == "SearchDatabaseByImage":
                    db_output = str(step[1])
                    
                    if "HIGH CONFIDENCE" in db_output or "MEDIUM CONFIDENCE" in db_output:
                        #  FIXED: Extract from numbered list format
                        parsed["destination_name"] = self._extract_destination_name(db_output)
                        parsed["district_name"] = self._extract_field(db_output, "District:")
                        parsed["category"] = self._extract_field(db_output, "Category:")
                        db_description = self._extract_field(db_output, "Description:")
                        
                        # Get visual analysis
                        visual_analysis = ""
                        for s in intermediate_steps:
                            if s[0].tool == "AnalyzeImageVisually":
                                visual_analysis = str(s[1])
                                break
                        
                        # Generate content using Gemini
                        logger.info(f"Generating content for DB match: {parsed['destination_name']}")
                        content_dict = self._generate_content_with_gemini(
                            destination_name=parsed["destination_name"],
                            district_name=parsed["district_name"],
                            category_name=parsed["category"],
                            db_description=db_description,
                            visual_analysis=visual_analysis,
                            gps_location=self._current_gps,
                            confidence="High" if "HIGH CONFIDENCE" in db_output else "Medium"
                        )
                        
                        # Update with generated content
                        parsed["historical_background"] = content_dict.get("historical_background", "")
                        parsed["cultural_significance"] = content_dict.get("cultural_significance", "")
                        parsed["what_makes_it_special"] = content_dict.get("what_makes_it_special", "")
                        parsed["visitor_experience"] = content_dict.get("visitor_experience", "")
                        parsed["interesting_facts"] = content_dict.get("interesting_facts", [])
                        
                        break
        
        return parsed
 
    def _extract_field(self, text: str, field_name: str) -> str:
        """Helper to extract single-line field value"""
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
        """Helper to extract destination name from database search output"""
        try:
            # Format: "1. Sigiriya [HIGH CONFIDENCE: 92%]"
            lines = db_output.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('1.'):
                    # Remove "1. " prefix and everything after "["
                    name = line[3:].split('[')[0].strip()
                    return name
            return "Unknown"
        except:
            return "Unknown"
    
    def _extract_multiline_field(self, text: str, start_marker: str, end_marker: str) -> str:
        """Helper to extract multi-line field content between two markers"""
        try:
            start_idx = text.find(start_marker)
            if start_idx == -1:
                return ""
            
            start_idx = text.find('\n', start_idx) + 1
            end_idx = text.find(end_marker, start_idx)
            
            if end_idx == -1:
                # If no end marker, take until next major section or end
                content = text[start_idx:].strip()
            else:
                content = text[start_idx:end_idx].strip()
            
            # Clean up the content
            return content.strip()
        except:
            return ""

    def _extract_bullet_points(self, text: str, marker: str) -> list:
        """Helper to extract bullet-pointed list"""
        try:
            start_idx = text.find(marker)
            if start_idx == -1:
                return []
            
            # Find the section after marker until "---" or end
            start_idx = text.find('\n', start_idx) + 1
            end_idx = text.find('---', start_idx)
            
            if end_idx == -1:
                section = text[start_idx:]
            else:
                section = text[start_idx:end_idx]
            
            # Extract lines that start with bullet points
            facts = []
            for line in section.split('\n'):
                line = line.strip()
                if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                    fact = line.lstrip('•-* ').strip()
                    if fact:
                        facts.append(fact)
            
            return facts
        except:
            return []

    


# Singleton instance
_agent_service = None

def get_agent_service() -> LocationIdentificationAgent:
    """Get or create LocationIdentificationAgent singleton"""
    global _agent_service
    if _agent_service is None:
        _agent_service = LocationIdentificationAgent()
    return _agent_service
    
    