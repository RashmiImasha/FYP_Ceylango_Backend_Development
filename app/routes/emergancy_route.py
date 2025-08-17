from fastapi import APIRouter, HTTPException, status, Query
from app.models.emergancy import EmergancyContact, EmergancyContactResponse, EmergancyNearest
from app.database.connection import emergancy_collection
from app.utils.destinationUtils import haversine
from typing import Union

router = APIRouter()

max_distance_km = 50

# add eContact data
@router.post("/", response_model=EmergancyContactResponse)
def add_eContact(contact: EmergancyContact):

    existing_contact = emergancy_collection.where('police_district', '==', contact.police_district ).get()
    if len(existing_contact) > 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"For the district '{contact.police_district}' Emergency contact already exists."            
        )
    
    doc_ref = emergancy_collection.document()
    data = contact.dict()
    data["id"] = doc_ref.id
    doc_ref.set(data)
    return {
        "message": "Emergency contact added successfully!",
        "data": EmergancyContact(**data)
    }

# updata eContact data
@router.put("/{contact_id}", response_model=EmergancyContactResponse)
def update_eContact(
    contact_id: str,
    updated_contact: EmergancyContact
):
    doc_ref = emergancy_collection.document(contact_id)
    if not doc_ref.get().exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contact details not found!"
        )
    
    existing_contact = emergancy_collection.where('police_district', '==', updated_contact.police_district ).get()
    for doc in existing_contact:
        if doc.id != contact_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"For the district '{updated_contact.police_district}' Emergency contact already exists."
            )       

    
    data = updated_contact.dict()
    data['id'] = contact_id
    doc_ref.update(data)    
    return {
        "message": "Emergency contact details updated successfully!",
        "data": updated_contact
    }

# get eContact data by Id
@router.get("/{contact_id}", response_model=EmergancyContact)
def get_eContact_byId( contact_id: str ):
    doc_ref = emergancy_collection.document(contact_id)
    eContact = doc_ref.get()

    if not eContact.exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Contact details not found!")
    
    eContact_details = eContact.to_dict()
    eContact_details['id'] = eContact.id
    return eContact_details
 
# get all eContact data
@router.get("/", response_model=list[EmergancyContact])
def get_all_eContact():
    eContacts = emergancy_collection.stream()
    result = []

    for doc in eContacts:
        data = doc.to_dict()
        data['id'] = doc.id
        result.append(data)
    return result

# delete eContact data
@router.delete("/{contact_id}")
def delete_eContact(contact_id: str):
    doc_ref = emergancy_collection.document(contact_id)
    eContact = doc_ref.get()

    if not eContact.exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Contact details not found!")
    
    doc_ref.delete()
    return {
        "message": "Destination deleted successfully",
        "contact_id": contact_id,
    }

# get eContact by user's location
@router.get("/eData/nearBy", response_model=Union[list[EmergancyNearest], EmergancyNearest])
def get_nearest_eContact(
    user_lat: float = Query(..., description="User's latitude"),
    user_lon: float = Query(..., description="User's longitude")    
):
    eContacts = emergancy_collection.stream()
    contact_within_range = []
    nearest_eContact = None
    min_distance = float("inf")

    for doc in eContacts:
        data = doc.to_dict()
        contact_lat = data["police_latitude"]
        contact_lon = data["police_longitude"]

        distance_m = haversine(user_lat, user_lon, contact_lat, contact_lon)
        distance_km = distance_m / 1000

        if distance_km < min_distance:
            min_distance = distance_km
            nearest_eContact = {**data, "id": doc.id, "distance_km":distance_km}
        
        if distance_km <= max_distance_km:
            contact_data = {**data, "id": doc.id, "distance_km":distance_km}
            contact_within_range.append(contact_data)

    if contact_within_range:
        contact_within_range.sort(key=lambda c: c["distance_km"])
        return contact_within_range
    elif nearest_eContact:
        return nearest_eContact
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUN, detail="No emergency contacts found")

