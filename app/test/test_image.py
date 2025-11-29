# app/scripts/generate_test_dataset.py
from app.database.connection import db
from firebase_admin import firestore
import random, json

def generate_test_dataset(test_size: int = 25, output_file: str = "test_dataset.json"):
    destination_ref = db.collection("destination")
    docs = destination_ref.stream()

    all_destinations = [doc.to_dict() | {"id": doc.id} for doc in docs]

    # Shuffle for randomness
    random.shuffle(all_destinations)

    test_dataset = all_destinations[:test_size]

    formatted_test_dataset = []

    for dest in test_dataset:
        formatted_test_dataset.append({
            "image_path": dest["destination_image"][0],  # first image
            "gps": {"lat": dest["latitude"], "lng": dest["longitude"]},
            "ground_truth_name": dest["destination_name"],
            "ground_truth_district": dest["district_name"],
            "difficulty": "easy",
            "notes": ""
        })

    with open(output_file, "w") as f:
        json.dump(formatted_test_dataset, f, indent=2)

    print(f"Test dataset saved to {output_file}")
    return formatted_test_dataset
