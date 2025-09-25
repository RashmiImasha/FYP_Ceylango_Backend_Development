from pydantic import BaseModel, EmailStr, Field
from typing import Literal, Optional, List, Dict, Any, Union
from enum import Enum

# Define main categories
class MainCategory(str, Enum):
    ACCOMMODATION = "accommodation"
    FOOD_DINING = "food_dining"
    WELLNESS = "wellness"
    SHOPPING = "shopping"
    ACTIVITIES = "activities"
    TRANSPORTATION = "transportation"

# Define subcategories for each main category
class AccommodationSubCategory(str, Enum):
    HOTEL = "hotel"
    GUESTHOUSE = "guesthouse"
    RESORT = "resort"
    VILLA = "villa"
    HOSTEL = "hostel"

class FoodDiningSubCategory(str, Enum):
    RESTAURANT = "restaurant"
    CAFE = "cafe"
    STREET_FOOD = "street_food"
    BAKERY = "bakery"
    BAR = "bar"

class WellnessSubCategory(str, Enum):
    AYURVEDIC_SPA = "ayurvedic_spa"
    MASSAGE_CENTER = "massage_center"
    YOGA_STUDIO = "yoga_studio"
    FITNESS_CENTER = "fitness_center"

class ShoppingSubCategory(str, Enum):
    SOUVENIR_SHOP = "souvenir_shop"
    LOCAL_MARKET = "local_market"
    BOUTIQUE = "boutique"
    JEWELRY_STORE = "jewelry_store"

class ActivitiesSubCategory(str, Enum):
    ADVENTURE_SPORTS = "adventure_sports"
    CULTURAL_SHOW = "cultural_show"
    TOUR_GUIDE = "tour_guide"
    WILDLIFE_SAFARI = "wildlife_safari"

class TransportationSubCategory(str, Enum):
    CAR_RENTAL = "car_rental"
    BOAT_SERVICE = "boat_service"
    TUK_TUK_SERVICE = "tuk_tuk_service"
    BICYCLE_RENTAL = "bicycle_rental"

# Base models
class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    role: Literal["admin", "tourist", "service_provider"] = "tourist"

class UserCreate(UserBase):
    password: str

class ServiceProviderApplication(BaseModel):
    email: EmailStr
    full_name: str
    service_name: str
    district: str
    main_category: MainCategory
    sub_category: str  # This will be validated based on main_category
    phone_number: str
    description: Optional[str] = None
    
    class Config:
        use_enum_values = True

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserInDB(UserBase):
    uid: str
    disabled: Optional[bool] = False
    main_category: Optional[MainCategory] = None
    sub_category: Optional[str] = None

# Service Provider Profile Models
class BaseServiceProfile(BaseModel):
    service_name: str
    description: str
    address: str
    district: str
    coordinates: Optional[Dict[str, float]] = None  # {"lat": 0.0, "lng": 0.0}
    phone_number: str
    email: Optional[EmailStr] = None
    website: Optional[str] = None
    social_media: Optional[Dict[str, str]] = None
    operating_hours: Dict[str, Dict[str, str]]  # {"monday": {"open": "09:00", "close": "17:00"}}
    images: List[str] = []  # URLs to images
    amenities: List[str] = []
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

# Category-specific profile extensions
class AccommodationProfile(BaseModel):
    room_types: List[Dict[str, Any]] = []
    check_in_time: str = "14:00"
    check_out_time: str = "11:00"
    cancellation_policy: Optional[str] = None
    room_amenities: List[str] = []
    hotel_amenities: List[str] = []
    price_range: Dict[str, float] = {}  # {"min": 1000, "max": 5000}

class FoodDiningProfile(BaseModel):
    cuisine_types: List[str] = []
    menu_items: List[Dict[str, Any]] = []
    dietary_options: List[str] = []  # ["vegetarian", "vegan", "gluten_free"]
    average_meal_price: Dict[str, float] = {}  # {"min": 500, "max": 2000}
    seating_capacity: Optional[int] = None
    delivery_available: bool = False
    takeaway_available: bool = False

class WellnessProfile(BaseModel):
    services_offered: List[Dict[str, Any]] = []
    therapists: List[Dict[str, str]] = []
    treatment_packages: List[Dict[str, Any]] = []
    booking_advance_days: int = 7
    cancellation_hours: int = 24

class ShoppingProfile(BaseModel):
    product_categories: List[str] = []
    inventory_items: List[Dict[str, Any]] = []
    payment_methods: List[str] = []
    shipping_available: bool = False
    return_policy: Optional[str] = None

class ActivitiesProfile(BaseModel):
    activity_types: List[str] = []
    duration: Optional[str] = None  # "2 hours", "Half day", "Full day"
    group_size: Dict[str, int] = {}  # {"min": 1, "max": 10}
    difficulty_level: Optional[str] = None  # "Easy", "Moderate", "Difficult"
    equipment_provided: List[str] = []
    age_restrictions: Optional[str] = None
    price_per_person: Optional[float] = None

class TransportationProfile(BaseModel):
    vehicle_types: List[Dict[str, Any]] = []
    coverage_areas: List[str] = []
    booking_advance_hours: int = 2
    driver_available: bool = True
    fuel_policy: Optional[str] = None
    insurance_included: bool = True

# Complete Service Provider Profile
class ServiceProviderProfile(BaseModel):
    uid: str
    main_category: MainCategory
    sub_category: str
    base_info: BaseServiceProfile
    category_data: Union[
        AccommodationProfile,
        FoodDiningProfile,
        WellnessProfile,
        ShoppingProfile,
        ActivitiesProfile,
        TransportationProfile,
        Dict[str, Any]  # For flexibility
    ]
    
    class Config:
        use_enum_values = True