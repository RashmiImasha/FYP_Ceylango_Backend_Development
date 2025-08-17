from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.category_route import router as category_route
from app.routes.auth_route import router as auth_route
from app.routes.destination_route import router as destination_route
from app.routes.image_route import router as image_route
from app.routes.emergancy_route import router as emergancy_route

# service_provider routes
from app.routes.service_providers.service_providers_route import router as service_providers_route
from app.routes.service_providers.event_route import router as event_route

# Create FastAPI app
app = FastAPI()

# CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

# register the routes
app.include_router(image_route, prefix="/image")
app.include_router(category_route, prefix="/category")
app.include_router(auth_route, prefix="/auth")
app.include_router(destination_route, prefix="/destination")
app.include_router(emergancy_route, prefix="/emergancy")

app.include_router(event_route, prefix="/event")
app.include_router(service_providers_route, prefix="/service_provider")

@app.get("/")
def root():
    return {"message": "Firestore backend is running"}
