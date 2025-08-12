import math
from PIL import Image
import imagehash

def haversine(lat1, lon1, lat2, lon2):
    # calculate distance between two points on the Earth ( Radius )
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi/2.0)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2.0)**2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def compute_phash(upload_file) -> str:
    image = Image.open(upload_file)
    phash = imagehash.phash(image)
    return str(phash)