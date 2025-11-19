import requests, re, logging
from app.database.connection import db
from app.config.settings import settings
import google.generativeai as genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class ServicesInitializer:
    
    def __init__(self):

        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.llm_model = settings.GEMINI_MODEL
        self.firestore_db = db
        self.OSRM_BASE_URL = settings.OSRM_URL      
        self.embedding_model = SentenceTransformer(settings.TEXT_EMBEDDING_MODEL)
        
        # Pinecone
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)

        # Configuration constants
        self.TOP_K_DESTINATIONS = 5
        self.TOP_K_SERVICES = 5
        self.TOP_K_EVENTS = 3
        self.RESPONSE_TEMPERATURE = 0.7
        self.MAX_RESPONSE_TOKENS = 200

        self.TEXT_NAMESPACE = 'destinationTextdata'  
        self.IMAGE_NAMESPACE = 'destinationImages' 
        
        logger.info(" All services initialized!")

class LanguageDetector:
    
    LANGUAGE_PATTERNS = {
        'si': r'[\u0D80-\u0DFF]',  # Sinhala Unicode range
        'ta': r'[\u0B80-\u0BFF]',  # Tamil Unicode range
        'hi': r'[\u0900-\u097F]',  # Hindi Unicode range
        'zh-CN': r'[\u4E00-\u9FFF]',  # Chinese Unicode range
        'ar': r'[\u0600-\u06FF]',  # Arabic Unicode range
    }
    
    LANGUAGE_NAMES = {
        'en': 'English',
        'si': 'Sinhala',
        'ta': 'Tamil',
        'hi': 'Hindi',
        'zh-CN': 'Chinese (Simplified)',
        'zh-TW': 'Chinese (Traditional)',
        'ar': 'Arabic',
        'de': 'German',
        'fr': 'French',
        'ja': 'Japanese',
        'ko': 'Korean',
        'es': 'Spanish',
        'ru': 'Russian'
    }
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect language from text"""
        # Check for non-Latin scripts
        for lang_code, pattern in LanguageDetector.LANGUAGE_PATTERNS.items():
            if re.search(pattern, text):
                return lang_code
        
        # Default to English
        return 'en'
    
    @staticmethod
    def get_language_name(code: str) -> str:
        """Get full language name"""
        return LanguageDetector.LANGUAGE_NAMES.get(code, 'English')

class QueryClassifier:
    
    SEMANTIC_KEYWORDS = [
        'romantic', 'peaceful', 'adventure', 'beautiful', 'authentic',
        'hidden', 'unique', 'best', 'amazing', 'stunning', 'tranquil',
        'exciting', 'relaxing', 'cultural', 'historical', 'scenic',
        'places', 'destinations', 'attractions', 'sights', 'visit', 
        'beach', 'beaches', 'sea', 'ocean', 'coast', 'shore', 
        'mountain', 'mountains', 'hill', 'forest', 'waterfall',
        'temple', 'temples', 'fort', 'fortress', 'historical site',        
    ]
    
    STRUCTURED_KEYWORDS = [
        'hotel', 'restaurant', 'shop', 'rental', 'car', 'bike',
        'event', 'open', 'near', 'price', 'cheap', 'expensive',
        'today', 'tonight', 'weekend', 'available',
        'service', 'services', 'food', 'stay', 'accommodation',          
    ]
    
    DIRECTIONS_KEYWORDS = [
        'how to go', 'how can i go', 'directions', 'route', 'way to',
        'how to reach', 'how to get', 'navigate', 'travel to', 'from', 'to',
        'from (.+) to (.+)', 
        'distance', 'how far', 'driving', 'walking',        
    ]
    
    LOCATION_BASED_KEYWORDS = [  
        'around', 'nearby', 'near me', 'close to', 'within',
        'nearest', 'closest',         
    ]
    
    @classmethod
    def classify(cls, query: str) -> str:
        """Classify query"""
        query_lower = query.lower()
    
        has_location_based = any(kw in query_lower for kw in cls.LOCATION_BASED_KEYWORDS)
        has_semantic = any(kw in query_lower for kw in cls.SEMANTIC_KEYWORDS)
        has_structured = any(kw in query_lower for kw in cls.STRUCTURED_KEYWORDS)
        has_directions = any(kw in query_lower for kw in cls.DIRECTIONS_KEYWORDS)
        
        # If query has location keywords (nearby, around) + destination keywords (beach, temple)
        # It should be SEMANTIC (search destinations), not location_based (search services)
        if has_location_based and has_semantic:
            return 'semantic'  # Search Pinecone for destinations
        
        if has_directions:
            return 'directions'
        elif has_location_based:
            return 'location_based'  # Only for services
        elif has_semantic and has_structured:
            return 'complex'
        elif has_semantic:
            return 'semantic'
        elif has_structured:
            return 'structured'
        else:
            return 'semantic'
        
class TranslationService:
    
    @staticmethod
    def translate_to_english(text: str, source_lang: str) -> str:

        if source_lang == 'en':
            return text
        
        try:
            translator = GoogleTranslator(source=source_lang, target='en')
            translation = translator.translate(text)
            logger.info(f"Translated: '{text}' → '{translation}'")
            return translation
        except Exception as e:
            logger.error(f" Translation failed: {e}, using original query")
            return text

class GeocodingService:
    
    NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
    
    @staticmethod
    def geocode(place_name: str, country: str = "Sri Lanka") -> Tuple[float, float]:
        """
        Convert place name to coordinates using Nominatim (FREE)
        Returns: (latitude, longitude) or None if not found
        """
        try:
            query = f"{place_name}, {country}"            
            response = requests.get(
                GeocodingService.NOMINATIM_URL,
                params={
                    "q": query,
                    "format": "json",
                    "limit": 1
                },
                headers={
                    "User-Agent": "TourismApp/1.0"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                results = response.json()
                if results:
                    lat = float(results[0]['lat'])
                    lon = float(results[0]['lon'])
                    logger.info(f" Geocoded '{place_name}': ({lat}, {lon})")
                    return (lat, lon)
            
            logger.error(f" GeocodingService: Could not geocode '{place_name}'")
            return None
            
        except Exception as e:
            logger.error(f" GeocodingService: Geocoding error: {e}")
            return None

class OSRMRoutingService:    
    
    @staticmethod
    def get_route(from_coords: Tuple[float, float], 
                  to_coords: Tuple[float, float],
                  mode: str = "driving",
                  osrm_base_url: str = None) -> Dict:
        
        """Get route from OSRM"""
        try:
            base_url = osrm_base_url 

            from_lon, from_lat = from_coords[1], from_coords[0]
            to_lon, to_lat = to_coords[1], to_coords[0]
            url = f"{base_url}/route/v1/{mode}/{from_lon},{from_lat};{to_lon},{to_lat}"
            
            response = requests.get(
                url,
                params={
                    "overview": "full",
                    "steps": "true",
                    "geometries": "geojson"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data['code'] == 'Ok' and data['routes']:
                    route = data['routes'][0]
                    
                    steps = []
                    if 'legs' in route:
                        for leg in route['legs']:
                            if 'steps' in leg:
                                for step in leg['steps']:
                                    if 'maneuver' in step:
                                        instruction = step['maneuver'].get('instruction', '')
                                        distance = step.get('distance', 0)
                                        steps.append({
                                            'instruction': instruction,
                                            'distance': distance
                                        })
                    
                    result = {
                        'distance': route['distance'],
                        'duration': route['duration'],
                        'steps': steps
                    }
                    
                    logger.info(f" OSRMRoutingService: Route found: {result['distance']/1000:.1f}km, {result['duration']/60:.0f} minutes")
                    return result
            
            logger.error(f"OSRMRoutingService: No route found")
            return None
            
        except Exception as e:
            logger.error(f"OSRMRoutingService: OSRM routing error: {e}")
            return None

class LocationExtractor:
    """Extract FROM and TO locations from directions query"""
    
    @staticmethod
    def extract_locations(query: str) -> Tuple[str, str]:
        """
        Extract source and destination from query
        Returns: (from_location, to_location)
        """
        query_lower = query.lower()
        
        patterns = [
            (r'from\s+(.+?)\s+to\s+(.+?)(?:\?|$)', 'en'),
            (r'how to go\s+(?:from\s+)?(.+?)\s+to\s+(.+?)(?:\?|$)', 'en'),
            (r'how can i go\s+(?:from\s+)?(.+?)\s+to\s+(.+?)(?:\?|$)', 'en'),
            (r'directions\s+(?:from\s+)?(.+?)\s+to\s+(.+?)(?:\?|$)', 'en'),
        ]
        
        for pattern, _ in patterns:
            match = re.search(pattern, query_lower)
            if match:
                from_loc = match.group(1).strip()
                to_loc = match.group(2).strip()
                logger.info(f" Extracted: FROM '{from_loc}' TO '{to_loc}'")
                return (from_loc, to_loc)
        
        if 'to ' in query_lower:
            parts = query_lower.split('to ')
            if len(parts) > 1:
                to_loc = parts[-1].strip().rstrip('?')
                return (None, to_loc)
        
        return (None, None)

class DirectionsHandler:
    """Handle directions queries using database + geocoding + OSRM"""
    
    def __init__(self, firebase_searcher, osrm_base_url: str):
        self.firebase_searcher = firebase_searcher
        self.osrm_base_url = osrm_base_url
    
    def get_coordinates(self, place_name: str, user_location: Tuple[float, float] = None) -> Tuple[float, float]:
        """
        Get coordinates for a place
        1. Check database first
        2. Use user location if place_name is None
        3. Fallback to geocoding
        """
        if not place_name:
            if user_location:
                logger.info(f"DirectionsHandler: Using user's current location: {user_location}")
                return user_location
            return None
        
        # Search database
        try:
            dest_query = self.firebase_searcher.db.collection('destination')\
                .where('destination_name', '==', place_name).limit(1).stream()
            
            for doc in dest_query:
                data = doc.to_dict()
                if 'latitude' in data and 'longitude' in data:
                    coords = (data['latitude'], data['longitude'])
                    logger.info(f"DirectionsHandler: Found '{place_name}' in database: {coords}")
                    return coords
            
            # Try case-insensitive
            all_dests = self.firebase_searcher.db.collection('destination').stream()
            for doc in all_dests:
                data = doc.to_dict()
                if data.get('destination_name', '').lower() == place_name.lower():
                    if 'latitude' in data and 'longitude' in data:
                        coords = (data['latitude'], data['longitude'])
                        logger.info(f"DirectionsHandler: Found '{place_name}' in database: {coords}")
                        return coords
        except Exception as e:
            logger.error(f"DirectionsHandler: Database search failed: {e}")
        
        # Fallback to geocoding
        logger.info(f"DirectionsHandler: Place not in database, using geocoding...")
        return GeocodingService.geocode(place_name)
    
    def handle_directions_query(self, query: str, user_location: Tuple[float, float] = None) -> Dict:
        """Handle a directions query"""
        
        from_place, to_place = LocationExtractor.extract_locations(query)
        
        if not to_place:
            return {'error': 'Could not understand the destination. Please specify where you want to go.'}
        
        from_coords = self.get_coordinates(from_place, user_location)
        to_coords = self.get_coordinates(to_place)
        
        if not from_coords:
            return {'error': 'Could not determine your starting location.'}
        
        if not to_coords:
            return {'error': f'Could not find location: {to_place}'}
        
        route = OSRMRoutingService.get_route(from_coords, to_coords, osrm_base_url=self.osrm_base_url)
        
        if not route:
            return {'error': 'Could not find a route.'}
        
        route['from_place'] = from_place or "Your location"
        route['to_place'] = to_place
        route['from_coords'] = from_coords
        route['to_coords'] = to_coords
        
        return route

class PineconeSearcher:
    """Search destinations in Pinecone using vector similarity"""
    
    def __init__(self, index, embedding_model, namespace: str): 
        self.index = index
        self.embedding_model = embedding_model
        self.namespace = namespace 
    
    def search(self, query: str, top_k: int = 5, filters: Dict = None) -> List[Dict]:
        """Search Pinecone for relevant destinations"""
        try:
            query_vector = self.embedding_model.encode(query).tolist()
            
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filters,
                namespace=self.namespace  
            )
            
            destinations = []
            for match in results['matches']:
                destinations.append({
                    'id': match['id'],
                    'score': match['score'],
                    'metadata': match['metadata']
                })
            
            logger.info(f"PineconeSearcher: found {len(destinations)} destinations in namespace '{self.namespace}'")
            return destinations
            
        except Exception as e:
            logger.error(f"PineconeSearcher: search error: {e}")
            return []
        
class FirebaseSearcher:
    """Search services in Firebase using structured queries"""
    
    def __init__(self, db):
        self.db = db
    
    def search_services(self, service_type: str = None, location: str = None, 
                       limit: int = 10) -> List[Dict]:  
        """
        Search services with flexible location matching
        """
        try:
            query = self.db.collection('service_provider_profiles')
            
            if service_type:
                query = query.where('service_category', '==', service_type)            
            if location:
                query = query.where('district', '==', location)
            
            query = query.where('is_active', '==', True)            
            results = query.limit(limit).stream()
            
            services = []
            for doc in results:
                data = doc.to_dict()
                data['id'] = doc.id
                services.append(data)
            
            logger.info(f"FirebaseSearcher: found {len(services)} services")
            return services
            
        except Exception as e:
            logger.error(f"FirebaseSearcher: search error: {e}")
            return []
    
    def search_services_near_location(self, lat: float, lon: float, 
                                      service_type: str = None, 
                                      radius_km: float = 10,
                                      limit: int = 10) -> List[Dict]:
        """
        Search services near coordinates using distance calculation
        Better for "around Galle" type queries
        """
        try:
            from math import radians, sin, cos, sqrt, atan2
            
            def calculate_distance(lat1, lon1, lat2, lon2):
                """Calculate distance in km using Haversine formula"""
                R = 6371  # in km
                
                dlat = radians(lat2 - lat1)
                dlon = radians(lon2 - lon1)
                
                a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                
                return R * c
            
            # Get all services (or filter by type first)
            query = self.db.collection('service_provider_profiles')
            
            if service_type:
                query = query.where('service_category', '==', service_type)
            
            query = query.where('is_active', '==', True)            
            all_services = query.stream()
            
            # Filter by distance
            nearby_services = []
            for doc in all_services:
                data = doc.to_dict()
                
                # Check if service has coordinates
                if data.get('coordinates'):
                    service_lat = data['coordinates'].get('lat')
                    service_lon = data['coordinates'].get('lng')
                    
                    if service_lat and service_lon:
                        distance = calculate_distance(lat, lon, service_lat, service_lon)
                        
                        if distance <= radius_km:
                            data['id'] = doc.id
                            data['distance_km'] = round(distance, 2)
                            nearby_services.append(data)
            
            # Sort by distance
            nearby_services.sort(key=lambda x: x['distance_km'])
            
            # Limit results
            nearby_services = nearby_services[:limit]
            
            logger.info(f"FirebaseSearcher: found {len(nearby_services)} services within {radius_km}km")
            return nearby_services
            
        except Exception as e:
            logger.error(f"FirebaseSearcher: location search error: {e}")
            return []   

class RetrievalOrchestrator:
    """Orchestrate retrieval from multiple sources including directions"""
    
    def __init__(self, pinecone_searcher, firebase_searcher, osrm_base_url: str):
        self.pinecone_searcher = pinecone_searcher
        self.firebase_searcher = firebase_searcher
        self.directions_handler = DirectionsHandler(firebase_searcher, osrm_base_url)
    
    def retrieve(self, query: str, query_type: str, 
                user_location: Tuple[float, float] = None,
                service_type: str = None,
                location: str = None) -> Dict:
        
        # Retrieve relevant data based on query type
        results = {
            'destinations': [],
            'services': [],
            'route': None
        }
        
        if query_type == 'directions':
            results['route'] = self.directions_handler.handle_directions_query(query, user_location)            
        elif query_type == 'semantic':
            results['destinations'] = self.pinecone_searcher.search(query, top_k=5)            
        elif query_type == 'structured':
            results['services'] = self.firebase_searcher.search_services(
                service_type=service_type,
                location=location,
                limit=10
            )            
        elif query_type == 'location_based':
            if user_location:
                results['services'] = self.firebase_searcher.search_services_near_location(
                    lat=user_location[0],
                    lon=user_location[1],
                    service_type=service_type,
                    radius_km=10,
                    limit=10
                )
            else:
                results['services'] = self.firebase_searcher.search_services(
                    service_type=service_type,
                    location=location,
                    limit=10
                )            
        elif query_type == 'complex':
            results['destinations'] = self.pinecone_searcher.search(query, top_k=3)
            
            if user_location:
                results['services'] = self.firebase_searcher.search_services_near_location(
                    lat=user_location[0],
                    lon=user_location[1],
                    service_type=service_type,
                    radius_km=10,
                    limit=5
                )
            else:
                results['services'] = self.firebase_searcher.search_services(
                    service_type=service_type,
                    location=location,
                    limit=5
                )
        
        # Enrich destinations with full data from Firebase
        for dest in results['destinations']:
            destination_id = dest['id']
            try:
                # Fetch full destination data from Firebase 'destination' collection
                doc = self.firebase_searcher.db.collection('destination').document(destination_id).get()
                if doc.exists:
                    dest['full_data'] = doc.to_dict()
                else:
                    dest['full_data'] = {}
            except Exception as e:
                logger.error(f"RetrievalOrchestrator: Error fetching destination {destination_id}: {e}")
                dest['full_data'] = {}
        
        return results

class ContextBuilder:
    """Build formatted context for LLM from retrieved data"""
    
    @staticmethod
    def build_context(results: Dict, query: str, language: str) -> str:

        context_parts = []

        # Add route information if it's a directions query
        if results.get('route'):
            route = results['route']
            
            if 'error' in route:
                return f"ERROR: {route['error']}"
            
            context_parts.append("ROUTE INFORMATION:")
            context_parts.append(f"From: {route.get('from_place', 'Your location')}")
            context_parts.append(f"To: {route.get('to_place', 'Destination')}")
            context_parts.append(f"Distance: {route['distance']/1000:.1f} km")
            context_parts.append(f"Duration: {route['duration']/60:.0f} minutes")
            
            if route.get('steps'):
                context_parts.append("\nTurn-by-turn directions:")
                for i, step in enumerate(route['steps'][:10], 1):
                    if step.get('instruction'):
                        dist = step.get('distance', 0)
                        context_parts.append(f"{i}. {step['instruction']} ({dist:.0f}m)")
            
            return "\n".join(context_parts)
        
        # Add destinations
        if results['destinations']:
            context_parts.append("DESTINATIONS:")
            for i, dest in enumerate(results['destinations'], 1):

                full_data = dest.get('full_data', {})                
                # Use full_data 
                name = full_data.get('destination_name') 
                category = full_data.get('category_name') 
                district = full_data.get('district_name') 
                description = full_data.get('description', 'N/A')
                rating = full_data.get('average_rating') 
                
                dest_info = f"""
                    {i}. {name}
                    Category: {category}
                    District: {district}
                    Description: {description[:300]}...
                    Rating: {rating}/5
                    """
                context_parts.append(dest_info)
        
        # Add services
        if results['services']:
            context_parts.append("\nSERVICES:")
            for i, service in enumerate(results['services'], 1):
                distance_info = f" ({service.get('distance_km')}km away)" if service.get('distance_km') else ""
                
                service_info = f"""
                    {i}. {service.get('service_name', 'Unknown')}{distance_info}
                    Category: {service.get('service_category', 'N/A')}
                    Location: {service.get('address', 'N/A')}
                    District: {service.get('district', 'N/A')}
                    Phone: {service.get('phone_number', 'N/A')}
                    Description: {service.get('description', 'N/A')[:200]}...
                    """
                context_parts.append(service_info)
        
        return "\n".join(context_parts)

class GeminiGenerator:
    """Generate response using Gemini API (in English), then translate"""
    
    @staticmethod
    def generate_response(query: str, context: str, language: str, 
                         query_type: str, 
                         gemini_model: str = settings.GEMINI_MODEL,
                         response_temperature: float = 0.7,
                         max_output_tokens: int = 500) -> str:
        
        if query_type == 'directions':
            system_instruction = """
                You are a helpful navigation assistant.
                Provide clear, easy-to-follow directions based on the route information provided.
                Be concise but include all important details like distance, duration, and key turns.
                Format the response naturally as if giving directions to a friend.
                Maximum 4-5 sentences total.
            """

        elif query_type == 'location_based':
            system_instruction = """
                You are a helpful tourism assistant for Sri Lanka.
                Provide information about services near the user's location.
                Include distance information when available.
                Be friendly and highlight the closest/most relevant options first.
                Maximum 4-5 services total.
            """

        elif query_type == 'structured':
            system_instruction = """
                You are a helpful tourism assistant for Sri Lanka.
                Provide information about services and places in the requested area.
                Be specific about locations and provide practical details like phone numbers and addresses when available.
                Maximum 3-4 sentences total.
            """

        elif query_type == 'semantic':
            system_instruction = """
                Answer questions about destinations and tourist attractions.
                Highlight the most interesting and relevant places based on what the user is looking for.
                Include ratings when available.
                Include rating. Maximum 60 words total.
            """

        elif query_type == 'complex':
            system_instruction = """
                You are a helpful tourism assistant for Sri Lanka.
                Provide comprehensive information about both destinations and nearby services.
                Organize your response to cover both tourist attractions and practical services.
                Be helpful and thorough.
                Keep total response under 75 words.
            """

        else:
            system_instruction = """
                You are a helpful tourism assistant for Sri Lanka. 
                Answer the user's question based ONLY on the provided information.
                If the information is not in the context, politely say you don't have that information.
                Give SHORT, helpful answers. Maximum 2-3 sentences.
                Be concise and friendly.
            """

        full_prompt = f"""{system_instruction}
            CONTEXT INFORMATION:
            {context}
            USER QUESTION: {query}
            Please provide a helpful answer in English:"""

        try:
            # Generate response in English
            model = genai.GenerativeModel(gemini_model)            
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=response_temperature,
                    max_output_tokens=max_output_tokens,
                )
            )
            
            english_answer = response.text.strip()
            logger.info(f"GeminiGenerator: response ({len(english_answer)} chars in English)")
            
            # Translate to user's language if not English
            if language != 'en':
                try:
                    translator = GoogleTranslator(source='en', target=language)
                    translated_answer = translator.translate(english_answer)
                    logger.info(f" Translated to {LanguageDetector.get_language_name(language)}")
                    return translated_answer
                except Exception as e:
                    logger.error(f"GeminiGenerator: Translation failed: {e}, returning English")
                    return english_answer
            
            return english_answer
            
        except Exception as e:
            logger.error(f"GeminiGenerator: Gemini generation error: {e}")
            return f"Sorry, I couldn't generate a response. Error: {str(e)}"

class MultilingualRAGChatbot:

    def __init__(self):

        services = ServicesInitializer()

        self.embedding_model = services.embedding_model
        self.pinecone_index = services.pinecone_index
        self.firestore_db = services.firestore_db
        self.gemini_model = services.llm_model

        self.pinecone_searcher = PineconeSearcher(
            index=self.pinecone_index,
            embedding_model=self.embedding_model,
            namespace=services.TEXT_NAMESPACE
        )

        self.firebase_searcher = FirebaseSearcher(self.firestore_db)
        self.retrieval = RetrievalOrchestrator(
            self.pinecone_searcher,
            self.firebase_searcher,
            osrm_base_url=services.OSRM_BASE_URL
        )
        logger.info("RAG Chatbot initialized and ready!")

    def chat(self, user_query: str, user_location: Tuple[float, float] = None,
            chat_history: List[Dict] = None) -> str:
        
        """
        Chat with history context support        
        Args:
            user_query: Current user question
            user_location: Optional user coordinates
            chat_history: Previous messages in format [{'role': 'user'/'assistant', 'content': '...'}]
        """

        lang = LanguageDetector.detect_language(user_query)

        # Translate to English
        translated_query = TranslationService.translate_to_english(user_query, lang)

        if chat_history and len(chat_history) > 0:
            # Get last 3 exchanges (6 messages) for context
            recent_history = chat_history[-6:]
            history_context = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in recent_history
            ])
            translated_query = f"Chat History:\n{history_context}\n\nCurrent Question: {translated_query}"

        query_type = QueryClassifier.classify(translated_query)

        # Retrieve documents / services / directions
        results = self.retrieval.retrieve(
            query=translated_query,
            query_type=query_type,
            user_location=user_location )

        # Build context for LLM
        context = ContextBuilder.build_context(results, translated_query, lang)

        # Generate answer using Gemini
        answer = GeminiGenerator.generate_response(
            query=translated_query,
            context=context,
            language=lang,
            query_type=query_type,
        )

        return answer


