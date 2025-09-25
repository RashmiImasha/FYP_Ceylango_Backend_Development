from typing import Dict, Any, List
from fastapi import HTTPException
from app.models.user import MainCategory

class ProfileValidator:
    """Validation utilities for service provider profiles"""
    
    @staticmethod
    def validate_accommodation_data(data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate accommodation-specific data"""
        errors = {}
        
        # Validate room types
        room_types = data.get("room_types", [])
        if not room_types:
            errors["room_types"] = ["At least one room type is required"]
        else:
            room_errors = []
            for i, room in enumerate(room_types):
                if not room.get("name"):
                    room_errors.append(f"Room {i+1}: Name is required")
                if not room.get("price_per_night") or room.get("price_per_night") <= 0:
                    room_errors.append(f"Room {i+1}: Valid price per night is required")
                if not room.get("max_occupancy") or room.get("max_occupancy") <= 0:
                    room_errors.append(f"Room {i+1}: Maximum occupancy must be at least 1")
            
            if room_errors:
                errors["room_types"] = room_errors
        
        # Validate check-in/check-out times
        if not data.get("check_in_time"):
            errors["check_in_time"] = ["Check-in time is required"]
        if not data.get("check_out_time"):
            errors["check_out_time"] = ["Check-out time is required"]
        
        # Validate price range
        price_range = data.get("price_range", {})
        if price_range:
            if price_range.get("min", 0) < 0:
                errors["price_range"] = ["Minimum price cannot be negative"]
            if price_range.get("max", 0) < price_range.get("min", 0):
                errors["price_range"] = ["Maximum price cannot be less than minimum price"]
        
        return errors
    
    @staticmethod
    def validate_food_dining_data(data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate food & dining specific data"""
        errors = {}
        
        # Validate menu items
        menu_items = data.get("menu_items", [])
        if not menu_items:
            errors["menu_items"] = ["At least one menu category with items is required"]
        else:
            menu_errors = []
            for i, category in enumerate(menu_items):
                if not category.get("category"):
                    menu_errors.append(f"Menu category {i+1}: Category name is required")
                
                items = category.get("items", [])
                if not items:
                    menu_errors.append(f"Menu category {i+1}: At least one item is required")
                else:
                    for j, item in enumerate(items):
                        if not item.get("name"):
                            menu_errors.append(f"Menu category {i+1}, Item {j+1}: Name is required")
                        if not item.get("price") or item.get("price") <= 0:
                            menu_errors.append(f"Menu category {i+1}, Item {j+1}: Valid price is required")
            
            if menu_errors:
                errors["menu_items"] = menu_errors
        
        # Validate seating capacity
        seating_capacity = data.get("seating_capacity")
        if seating_capacity is not None and seating_capacity <= 0:
            errors["seating_capacity"] = ["Seating capacity must be greater than 0"]
        
        return errors
    
    @staticmethod
    def validate_wellness_data(data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate wellness service specific data"""
        errors = {}
        
        # Validate services offered
        services = data.get("services_offered", [])
        if not services:
            errors["services_offered"] = ["At least one service is required"]
        else:
            service_errors = []
            for i, service in enumerate(services):
                if not service.get("service_name"):
                    service_errors.append(f"Service {i+1}: Service name is required")
                if not service.get("duration_minutes") or service.get("duration_minutes") <= 0:
                    service_errors.append(f"Service {i+1}: Valid duration is required")
                if not service.get("price") or service.get("price") <= 0:
                    service_errors.append(f"Service {i+1}: Valid price is required")
            
            if service_errors:
                errors["services_offered"] = service_errors
        
        # Validate booking settings
        if data.get("booking_advance_days", 0) < 0:
            errors["booking_advance_days"] = ["Booking advance days cannot be negative"]
        if data.get("cancellation_hours", 0) < 0:
            errors["cancellation_hours"] = ["Cancellation hours cannot be negative"]
        
        return errors
    
    @staticmethod
    def validate_shopping_data(data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate shopping service specific data"""
        errors = {}
        
        # Validate inventory items
        inventory = data.get("inventory_items", [])
        if not inventory:
            errors["inventory_items"] = ["At least one product is required"]
        else:
            inventory_errors = []
            for i, item in enumerate(inventory):
                if not item.get("product_name"):
                    inventory_errors.append(f"Product {i+1}: Product name is required")
                if not item.get("price") or item.get("price") <= 0:
                    inventory_errors.append(f"Product {i+1}: Valid price is required")
                if item.get("stock_quantity", 0) < 0:
                    inventory_errors.append(f"Product {i+1}: Stock quantity cannot be negative")
            
            if inventory_errors:
                errors["inventory_items"] = inventory_errors
        
        # Validate product categories
        if not data.get("product_categories"):
            errors["product_categories"] = ["At least one product category is required"]
        
        return errors
    
    @staticmethod
    def validate_activities_data(data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate activities service specific data"""
        errors = {}
        
        # Validate activity types
        if not data.get("activity_types"):
            errors["activity_types"] = ["At least one activity type is required"]
        
        # Validate group size
        group_size = data.get("group_size", {})
        if group_size:
            min_size = group_size.get("min", 1)
            max_size = group_size.get("max", 1)
            
            if min_size <= 0:
                errors["group_size"] = ["Minimum group size must be at least 1"]
            elif max_size < min_size:
                errors["group_size"] = ["Maximum group size cannot be less than minimum"]
        
        # Validate price
        price = data.get("price_per_person")
        if price is not None and price <= 0:
            errors["price_per_person"] = ["Price per person must be greater than 0"]
        
        return errors
    
    @staticmethod
    def validate_transportation_data(data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate transportation service specific data"""
        errors = {}
        
        # Validate vehicle types
        vehicles = data.get("vehicle_types", [])
        if not vehicles:
            errors["vehicle_types"] = ["At least one vehicle type is required"]
        else:
            vehicle_errors = []
            for i, vehicle in enumerate(vehicles):
                if not vehicle.get("type"):
                    vehicle_errors.append(f"Vehicle {i+1}: Vehicle type is required")
                if not vehicle.get("capacity") or vehicle.get("capacity") <= 0:
                    vehicle_errors.append(f"Vehicle {i+1}: Valid capacity is required")
                
                # Validate pricing
                day_price = vehicle.get("price_per_day")
                km_price = vehicle.get("price_per_km")
                if not day_price and not km_price:
                    vehicle_errors.append(f"Vehicle {i+1}: Either daily or per-km pricing is required")
                elif day_price and day_price <= 0:
                    vehicle_errors.append(f"Vehicle {i+1}: Daily price must be greater than 0")
                elif km_price and km_price <= 0:
                    vehicle_errors.append(f"Vehicle {i+1}: Per-km price must be greater than 0")
            
            if vehicle_errors:
                errors["vehicle_types"] = vehicle_errors
        
        # Validate booking advance hours
        if data.get("booking_advance_hours", 0) < 0:
            errors["booking_advance_hours"] = ["Booking advance hours cannot be negative"]
        
        return errors
    
    @classmethod
    def validate_category_data(cls, main_category: str, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Main validation method that routes to category-specific validators"""
        validators = {
            MainCategory.ACCOMMODATION: cls.validate_accommodation_data,
            MainCategory.FOOD_DINING: cls.validate_food_dining_data,
            MainCategory.WELLNESS: cls.validate_wellness_data,
            MainCategory.SHOPPING: cls.validate_shopping_data,
            MainCategory.ACTIVITIES: cls.validate_activities_data,
            MainCategory.TRANSPORTATION: cls.validate_transportation_data,
        }
        
        validator = validators.get(main_category)
        if validator:
            return validator(data)
        
        return {}
    
    @staticmethod
    def validate_base_info(data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate base profile information"""
        errors = {}
        
        required_fields = ["service_name", "description", "address", "district", "phone_number"]
        for field in required_fields:
            if not data.get(field):
                errors[field] = [f"{field.replace('_', ' ').title()} is required"]
        
        # Validate phone number format (basic validation)
        phone = data.get("phone_number")
        if phone and not phone.replace("+", "").replace("-", "").replace(" ", "").isdigit():
            errors["phone_number"] = ["Invalid phone number format"]
        
        # Validate operating hours
        operating_hours = data.get("operating_hours", {})
        if operating_hours:
            days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            hours_errors = []
            
            for day in days:
                day_hours = operating_hours.get(day, {})
                if day_hours:
                    open_time = day_hours.get("open")
                    close_time = day_hours.get("close")
                    
                    if not open_time or not close_time:
                        hours_errors.append(f"{day.title()}: Both opening and closing times are required")
                    # Add time format validation here if needed
            
            if hours_errors:
                errors["operating_hours"] = hours_errors
        
        return errors