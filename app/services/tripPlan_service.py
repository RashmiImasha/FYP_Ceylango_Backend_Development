from firebase_admin import firestore
from app.config.settings import settings
from app.database.connection import db, tripPlan_collection, destination_collection
import google.generativeai as genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from app.utils.tripPlan_utils import get_osrm_route, get_visit_duration, get_destination_categories_for_interests
from app.models.tripPlan import TripPlanRequest
from datetime import datetime
from typing import List, Dict
import logging, asyncio, json
import numpy as np

logger = logging.getLogger(__name__)

def generate_trip_name(districts: List[str], start_date: str, interests: List[str], group_type: str) -> str:

    date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    month_year = date_obj.strftime("%b %Y")
    
    interest_map = {
        'adventure sports': 'Adventure',
        'cultural sites': 'Cultural',
        # 'food and dining': 'Culinary',
        'nature and wildlife': 'Nature',
        'museums and art': 'Art & Culture',
        'beaches and relaxation': 'Beach',
        # 'shopping': 'Shopping',
        'nightlife': 'Nightlife',
        'photography': 'Photography',
        'local experience': 'Local',
        'historical places': 'Heritage',
        # 'wellness and spa': 'Wellness'
    }
    
    primary_interest = interests[0] if interests else 'Discovery'
    interest_label = interest_map.get(primary_interest.lower(), primary_interest.title())
    
    group_descriptors = {
        'solo': 'Solo',
        'couple': 'Romantic',
        'family': 'Family',
        'friends': 'Friends'
    }
    
    group_label = group_descriptors.get(group_type.lower(), '')
    
    if len(districts) == 1:
        location = districts[0]
    elif len(districts) == 2:
        location = f"{districts[0]} & {districts[1]}"
    else:
        location = f"{districts[0]} & {len(districts)-1} more"
    
    if group_type == 'couple':
        name = f"{group_label} {interest_label} in {location} - {month_year}"
    elif group_type == 'solo':
        name = f"{group_label} {interest_label} Journey - {location} - {month_year}"
    else:
        name = f"{location} {group_label} {interest_label} - {month_year}"
    
    return name

class ServicesInitializer:
    
    def __init__(self):
        self.firestore_db = db
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)
        self.TEXT_NAMESPACE = 'destinationTextdata' 
        self.OSRM_BASE_URL = settings.OSRM_URL 
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.llm_model = genai.GenerativeModel(settings.GEMINI_MODEL)
        self.embedding_model = SentenceTransformer(settings.TEXT_EMBEDDING_MODEL)

class DestinationRetriever:

    def __init__(self, services):
        self.pinecone_index = services.pinecone_index
        self.firestore_db = services.firestore_db
        self.embedding_model = services.embedding_model
        self.namespace = services.TEXT_NAMESPACE

    async def generate_query_embedding(self, interests: List[str], group_type: str) -> List[float]:
        query_text = f"{' '.join(interests)} travel {group_type} tourism trip vacation"
        try:
            embedding = self.embedding_model.encode(query_text, convert_to_numpy=True)
            return embedding.tolist()

        except Exception as e:
            logger.error(f"[Embedding ERROR] {e}")
            dim = self.embedding_model.get_sentence_embedding_dimension()
            return [0.0] * dim

    async def query_pinecone(self, embedding, districts: List[str], categories: List[str] = None, top_k=30):
        """
        Query Pinecone with district and optional category filters
        """
        filter_dict = {}

        if districts:
            normalized_districts = [d.strip().title() for d in districts]
            filter_dict["district_name"] = {"$in": normalized_districts}
        
        if categories:
            filter_dict["category_name"] = {"$in": categories}

        try:
            results = self.pinecone_index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=self.namespace,
                filter=filter_dict if filter_dict else None
            )            
            return [
                {
                    "destination_id": m["id"],
                    "similarity_score": m["score"],
                    "metadata": m.get("metadata", {})
                }
                for m in results.get("matches", [])
            ]

        except Exception as e:
            logger.error(f"[Pinecone Query ERROR] {e}")
            logger.error(f"Filter used: {filter_dict}")
            return []

    async def enrich_with_firebase(self, destination_ids: List[str]) -> List[Dict]:
        """
        Fetch full destination data from Firebase using imported collection
        """        
        try:
            destinations = []
            batch_size = 10  # Firestore 'in' query limit

            for i in range(0, len(destination_ids), batch_size):
                batch_ids = destination_ids[i:i + batch_size]

                docs = (destination_collection.where("__name__", "in", batch_ids).get())
                for doc in docs:
                    obj = doc.to_dict()
                    obj["destination_id"] = doc.id
                    obj["item_type"] = "destination"
                    destinations.append(obj)
            
            # logger.info(f"Firebase returned {len(destinations)} destinations")
            return destinations

        except Exception as e:
            logger.error(f"[Firebase Fetch ERROR] {e}")
            return []

    async def retrieve_destinations(self, interests, districts, group_type):
        """
        MAIN PIPELINE:
        1. Create semantic query from interests
        2. Pinecone similarity search
        3. Enrich with full Firebase data
        """        
        # Map interests to categories
        categories = get_destination_categories_for_interests(interests)
        logger.info(f"Mapped interests {interests} to categories: {categories}")

        # Step 1: Query embedding
        embedding = await self.generate_query_embedding(interests, group_type)

        # Step 2: Pinecone similarity search with categories
        pinecone_results = await self.query_pinecone(embedding, districts, categories)
        logger.info(f"Found {len(pinecone_results)} destinations from Pinecone")

        if not pinecone_results:
            # Fallback: try without category filter
            pinecone_results = await self.query_pinecone(embedding, districts, categories=None)
            logger.info(f"Without category filter Pinecone returned {len(pinecone_results)} results")

        if not pinecone_results:
            return []

        destination_ids = [r["destination_id"] for r in pinecone_results]

        # Step 3: Firebase enrichment
        firebase_data = await self.enrich_with_firebase(destination_ids)
        fb_map = {item["destination_id"]: item for item in firebase_data}

        final_results = []
        for result in pinecone_results:
            did = result["destination_id"]
            if did in fb_map:
                item = fb_map[did]
                item["similarity_score"] = result["similarity_score"]
                final_results.append(item)        
        
        logger.info(f"Returning {len(final_results)} enriched destinations")        
        return final_results
    
class TripPreprocessor:
    """Handles distance calculations and business rules"""
    
    def __init__(self, services: ServicesInitializer):        
        self.osrm_url = services.OSRM_BASE_URL
        self.embedding_model = services.embedding_model
        self._category_cache = {}
    
    async def calculate_distance_and_time_matrices(self, items: List[Dict]) -> tuple:

        n = len(items)
        distance_matrix = np.zeros((n, n))
        time_matrix = np.zeros((n, n))
        
        tasks = []
        for i in range(n):
            for j in range(i+1, n):
                task = get_osrm_route(
                    items[i]['latitude'],
                    items[i]['longitude'],
                    items[j]['latitude'],
                    items[j]['longitude']
                )
                tasks.append((i, j, task))
        
        if tasks:
            logger.info(f"TripPreprocessor: Calculating {len(tasks)} routes with OSRM...")
            results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)
            
            for (i, j, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    logger.error(f"TripPreprocessor: OSRM failed for pair ({i},{j}): {result}")
                    distance_matrix[i][j] = 50.0
                    distance_matrix[j][i] = 50.0
                    time_matrix[i][j] = 60
                    time_matrix[j][i] = 60
                else:
                    distance_matrix[i][j] = result['distance_km']
                    distance_matrix[j][i] = result['distance_km']
                    time_matrix[i][j] = result['duration_mins']
                    time_matrix[j][i] = result['duration_mins']
        
        return distance_matrix, time_matrix
    
    def get_or_create_category_embedding(self, category_name: str) -> np.ndarray:
        """Cache category embeddings"""
        if category_name not in self._category_cache:
            self._category_cache[category_name] = self.embedding_model.encode(
                category_name, convert_to_numpy=True
            )
        return self._category_cache[category_name]
    
    def calculate_category_interest_score(self, category_name: str, interest_embeddings: np.ndarray) -> float:
        """Calculate semantic similarity"""
        category_embedding = self.get_or_create_category_embedding(category_name)
        
        similarities = []
        for interest_emb in interest_embeddings:
            similarity = np.dot(category_embedding, interest_emb) / (
                np.linalg.norm(category_embedding) * np.linalg.norm(interest_emb)
            )
            similarities.append(similarity)
        
        return float(np.max(similarities))
    
    async def apply_business_rules(
        self, 
        destinations: List[Dict],
        group_type: str, 
        interests: List[str]
    ) -> List[Dict]:
        """Filter and score destinations """

        interest_embeddings = self.embedding_model.encode(interests, convert_to_numpy=True)        
        all_items = []
        
        # Process destinations
        for dest in destinations:
            rating = dest.get('average_rating', 0)
            if rating < 3.5:
                continue            
            category = dest.get('category_name', '').lower()
            
            # Family-friendly filter
            if group_type == "family":
                blocked_keywords = ['nightlife', 'bar', 'club', 'casino', 'adult']
                if any(keyword in category for keyword in blocked_keywords):
                    continue
            
            # Vector-based interest matching
            interest_score = self.calculate_category_interest_score(
                category_name=dest.get('category_name', 'general'),
                interest_embeddings=interest_embeddings
            )
            dest['interest_match_score'] = interest_score
            
            # Couple boost
            couple_boost = 0.0
            if group_type == "couple":
                romantic_keywords = ['beach', 'sunset', 'nature', 'scenic', 'spa']
                if any(keyword in category for keyword in romantic_keywords):
                    couple_boost = 0.15
            dest['couple_boost'] = couple_boost
            
            # Visit duration
            dest['typical_visit_duration'] = get_visit_duration(
                dest.get('category_name', 'historical'),
                group_type
            )
            
            # Composite score
            composite_score = (
                dest.get('similarity_score', 0) * 0.40 +
                interest_score * 0.35 +
                (rating / 5.0) * 0.15 +
                couple_boost * 0.10
            )
            dest['composite_score'] = composite_score
            
            if composite_score >= 0.3:
                all_items.append(dest)
        
        # Sort by composite score
        all_items.sort(key=lambda x: x['composite_score'], reverse=True)
        logger.info(f"Filtered {len(all_items)} destinations")
        return all_items

class TripPlanningAgent:
    """AI Agent for itinerary planning"""
    
    def __init__(self, services):        
        self.llm_model = services.llm_model
    
    def construct_planning_prompt(
        self,
        items: List[Dict],
        travel_time_matrix: np.ndarray,
        districts: List[str],
        start_date: str,
        end_date: str,
        group_type: str,
        interests: List[str],
        transport_mode: str = "car"
    ) -> str:
        duration_days = (datetime.strptime(end_date, "%Y-%m-%d") - 
                        datetime.strptime(start_date, "%Y-%m-%d")).days + 1
        
        daily_hours = 10
        total_minutes = duration_days * daily_hours * 60
        meal_time = duration_days * 180
        buffer_time = int(total_minutes * 0.15)
        usable_time = total_minutes - meal_time - buffer_time
        
        # Prepare items list (destinations + services)
        item_list = []
        for idx, item in enumerate(items[:25]):  # Increased to 25 for services
            item_data = {
                "index": idx,
                "type": item.get('item_type'),
                "id": item.get('destination_id') or item.get('service_id'),
                "name": item.get('destination_name') or item.get('service_name'),
                "category": item.get('category_name') or item.get('service_category'),
                "district": item.get('district_name') or item.get('district'),
                "rating": item.get('average_rating', 0),
                "visit_duration_mins": item['typical_visit_duration'],
                "latitude": item['latitude'],
                "longitude": item['longitude']
            }
            if item.get('item_type') == 'destination':
                item_data["description"] = item.get('description', '')[:200]
            item_list.append(item_data)
        
        # Travel times
        travel_times = []
        for i in range(min(25, len(items))):
            for j in range(i+1, min(25, len(items))):
                travel_times.append({
                    "from_idx": i,
                    "to_idx": j,
                    "time_mins": int(travel_time_matrix[i][j])
                })

        prompt = f"""You are an expert travel planner for Sri Lanka. Create an optimal {duration_days}-day itinerary.

            **USER REQUIREMENTS:**
            - Districts: {', '.join(districts)}
            - Duration: {duration_days} days (from {start_date} to {end_date})
            - Group Type: {group_type}
            - Interests: {', '.join(interests)}
            - Transport: {transport_mode}

            **TIME CONSTRAINTS:**
            - Total available time: {usable_time} minutes ({usable_time/60:.1f} hours)
            - Daily active hours: {daily_hours} hours (8:00 AM - 6:00 PM)

            **AVAILABLE DESTINATIONS:**
            {json.dumps(item_list, indent=2)}

            **TRAVEL TIMES (minutes between places):**
            {json.dumps(travel_times[:60], indent=2)}

            **CRITICAL RULES:**
            1. DO NOT create separate "travel" activities - travel time is automatically included in "travel_from_previous_mins"
            2. Each place should appear ONLY ONCE as a "visit" activity
            3. Never revisit the same location unless it makes logical sense (e.g., hotel)
            4. Select diverse types of destinations
            5. Select 4-5 UNIQUE places per day
            6. Minimize travel time by clustering nearby places
            7. Start at 8:00 AM, end by 6:00 PM.
            8. Include helpful "notes" for each activity with what to see/do and special tips

            **ACTIVITY TYPES:**
            - "visit": For tourist destinations (temples, forts, beaches)            
            - "activity": For adventure sports, spa, etc.
            - DO NOT use "travel" or "break" types - these are handled automatically

            **OUTPUT FORMAT (strictly JSON):**
            {{
            "reasoning": "Brief explanation of planning strategy",
            "itinerary": [
                {{
                "day": 1,
                "date": "{start_date}",
                "theme": "Cultural Exploration & Local Cuisine",
                "activities": [
                    {{
                    "start_time": "08:00",
                    "end_time": "10:00",
                    "item_index": 0,
                    "activity_type": "visit",
                    "travel_from_previous_mins": 0,
                    "notes": "Morning visit to explore architecture"
                    }},
                    {{
                    "start_time": "10:30",
                    "end_time": "12:00",
                    "item_index": 2,
                    "activity_type": "visit",
                    "travel_from_previous_mins": 15,
                    "notes": "Scenic viewpoint"
                    }},
                    {{
                    "start_time": "12:30",
                    "end_time": "13:30",
                    "item_index": 5,
                    "activity_type": "meal",
                    "travel_from_previous_mins": 20,
                    "notes": "Lunch at local restaurant"
                    }}
                ]
                }}
            ],
            "summary": {{
                "total_destinations": 8,                
                "total_travel_time_mins": 240,
                "total_distance_km": 120
            }}
            }}

            REMEMBER: Each location appears only once with its activity type. Travel time is in "travel_from_previous_mins" field!"""
                    
        return prompt
    
    async def generate_itinerary(
        self,
        items: List[Dict],
        travel_time_matrix: np.ndarray,
        trip_request: Dict
    ) -> Dict:
        prompt = self.construct_planning_prompt(
            items=items,
            travel_time_matrix=travel_time_matrix,
            districts=trip_request.get('districts'),
            start_date=trip_request.get('start_date'),
            end_date=trip_request.get('end_date'),
            group_type=trip_request.get('group_type'),
            interests=trip_request.get('interests'),
            transport_mode=trip_request.get('transport_mode', 'car')
        )
        
        try:
            response = self.llm_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=4096,
                )
            )            
            response_text = response.text
            
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            itinerary_data = json.loads(response_text.strip())
            return itinerary_data
            
        except Exception as e:
            logger.error(f"TripPlanningAgent: Error generating itinerary: {e}")
            raise Exception(f"Failed to generate itinerary: {str(e)}")

class ItineraryPostProcessor:
    
    @staticmethod
    def validate_itinerary(itinerary_data: Dict, request: TripPlanRequest) -> bool:
        duration_days = (datetime.strptime(request.end_date, "%Y-%m-%d") - 
                        datetime.strptime(request.start_date, "%Y-%m-%d")).days + 1
        
        if len(itinerary_data.get('itinerary', [])) != duration_days:
            return False
        
        for day in itinerary_data['itinerary']:
            if not day.get('activities'):
                return False
        
        return True
    
    @staticmethod
    def enhance_itinerary(
        itinerary_data: Dict,
        items: List[Dict],
        distance_matrix: np.ndarray
    ) -> Dict:
        item_lookup = {i: item for i, item in enumerate(items)}
        enhanced_itinerary = []
        
        for day in itinerary_data['itinerary']:
            enhanced_day = {
                "day_number": day['day'],
                "date": day['date'],
                "theme": day.get('theme', 'Exploration'),
                "activities": [],
                "total_distance_km": 0,
                "total_travel_time_mins": 0
            }

            seen_items = set()  # Track visited locations
            prev_item_idx = None
            
            for activity in day['activities']:
                                
                # Skip "travel" activities
                if activity.get('activity_type') == 'travel':
                    continue
                item_idx = activity.get('item_index')
                # Skip duplicate visits to same location
                if item_idx is not None and item_idx in seen_items:
                    continue          

                if item_idx is not None:
                    seen_items.add(item_idx)  
            
                enhanced_activity = {
                    "start_time": activity['start_time'],
                    "end_time": activity['end_time'],
                    "activity_type": activity['activity_type'],
                    "duration_mins": 0,
                    "travel_from_previous_mins": activity.get('travel_from_previous_mins', 0),
                }
                
                start = datetime.strptime(activity['start_time'], "%H:%M")
                end = datetime.strptime(activity['end_time'], "%H:%M")
                enhanced_activity['duration_mins'] = int((end - start).seconds / 60)
                
                if item_idx is not None and item_idx in item_lookup:
                    item = item_lookup[item_idx]
                    
                    enhanced_activity.update({
                        "destination_id": item['destination_id'],
                        "destination_name": item['destination_name'],
                        # "category": item['category_name'],
                        # "description": item.get('description', '')[:200],
                        "rating": item.get('average_rating', 0),
                        "coordinates": {
                            "latitude": item['latitude'],
                            "longitude": item['longitude']
                        },
                        "tips": activity.get('notes', '')
                    })
                    
                    if prev_item_idx is not None:
                        distance = distance_matrix[prev_item_idx][item_idx]
                        enhanced_day['total_distance_km'] += distance
                    
                    prev_item_idx = item_idx
                
                enhanced_day['activities'].append(enhanced_activity)
            
            enhanced_itinerary.append(enhanced_day)
        
        return {
            "itinerary": enhanced_itinerary,
            "summary": itinerary_data.get('summary', {}),
            "reasoning": itinerary_data.get('reasoning', '')
        }
    
    @staticmethod
    def generate_map_data(enhanced_itinerary: Dict) -> Dict:
        route_points = []
        markers = []
        
        for day in enhanced_itinerary['itinerary']:
            for activity in day['activities']:
                coords = activity.get('coordinates', {})
                if coords:
                    name = activity.get('destination_name') or activity.get('service_name', 'Location')
                    point = {
                        "lat": coords['latitude'],
                        "lng": coords['longitude'],
                        "name": name,
                        "day": day['day_number'],
                        "time": activity['start_time']
                    }
                    route_points.append(point)
                    markers.append(point)
        
        return {
            "route_points": route_points,
            "markers": markers,
            "center": route_points[0] if route_points else {"lat": 7.8731, "lng": 80.7718}
        }

async def save_trip_plan(trip_id: str, trip_name: str, request: TripPlanRequest, response: Dict):
    try:
        trip_data = {
            "trip_id": trip_id,
            "trip_name": trip_name,
            "user_id": request.user_id,
            "created_at": firestore.SERVER_TIMESTAMP,
            "generated_at": response.get('generated_at', datetime.now().isoformat()),
            "request": request.dict(),
            "itinerary": response['itinerary'],
            "summary": response['summary'],
            "map_data": response['map_data'],            
        }
        
        tripPlan_collection.document(trip_id).set(trip_data)
        logger.info(f"Trip plan {trip_id} saved with name: {trip_name}")
        
    except Exception as e:
        logger.error(f"Error saving trip plan: {e}")
        raise

async def generate_trip_plan(request: TripPlanRequest):
    """Main function to generate trip plan"""
    services = ServicesInitializer()
    
    # Step 1: Retrieve destinations (RAG with Pinecone)
    dest_retriever = DestinationRetriever(services)
    destinations = await dest_retriever.retrieve_destinations(
        interests=request.interests,
        districts=request.districts,
        group_type=request.group_type
    )
    
    if not destinations:
        raise Exception("No destinations found matching your criteria")
    
    # Step 3: Apply business rules to both
    preprocessor = TripPreprocessor(services)
    all_items = await preprocessor.apply_business_rules(
        destinations=destinations,
        group_type=request.group_type,
        interests=request.interests
    )
    
    if not all_items:
        raise Exception("No suitable places after filtering")
    
    # Step 4: Calculate distances
    distance_matrix, travel_time_matrix = await preprocessor.calculate_distance_and_time_matrices(all_items)
    
    # Step 5: Generate itinerary with AI
    agent = TripPlanningAgent(services)
    trip_request_dict = {
        'districts': request.districts,
        'start_date': request.start_date,
        'end_date': request.end_date,
        'group_type': request.group_type,
        'interests': request.interests,
        'transport_mode': request.transport_mode
    }
    
    itinerary_data = await agent.generate_itinerary(
        items=all_items,
        travel_time_matrix=travel_time_matrix,
        trip_request=trip_request_dict
    )
    
    # Step 6: Validate
    is_valid = ItineraryPostProcessor.validate_itinerary(itinerary_data, request)
    
    if not is_valid:
        raise Exception("Generated itinerary failed validation")
    
    # Step 7: Enhance
    enhanced_data = ItineraryPostProcessor.enhance_itinerary(
        itinerary_data=itinerary_data,
        items=all_items,
        distance_matrix=distance_matrix
    )
    
    # Step 8: Generate map data
    map_data = ItineraryPostProcessor.generate_map_data(enhanced_data)

    # Step 9: Generate trip name
    trip_name = generate_trip_name(
        districts=request.districts,
        start_date=request.start_date,
        interests=request.interests,
        group_type=request.group_type
    )
    
    # Step 10: Prepare response
    trip_id = f"trip_{request.user_id}_{int(datetime.now().timestamp())}"
    
    response = {
        "trip_id": trip_id,
        "trip_name": trip_name,        
        "summary": enhanced_data['summary'],
        "itinerary": enhanced_data['itinerary'],
        "map_data": map_data,
        "reasoning": enhanced_data['reasoning'],
        "generated_at": datetime.now().isoformat()
    }
    
    # Step 11: Save to database
    await save_trip_plan(trip_id, trip_name, request, response)       
    return response



