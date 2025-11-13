import uvicorn, logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.chat_route import router as chat_route
from app.routes.category_route import router as category_route
from app.routes.auth_route import router as auth_route
from app.routes.destination_route import router as destination_route
from app.routes.image_route import router as image_route
from app.routes.emergancy_route import router as emergancy_route

# service_provider routes
from app.routes.service_providers.service_providers_route import router as service_providers_route
from app.routes.service_providers.service_provider_profile_route import router as service_provider_profile_route
from app.routes.service_providers.event_route import router as event_route
from app.routes.review_routes import router as review_route
from app.routes.popular_toprated_routes import router as popular_toprated_route

logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Print to console
        logging.FileHandler('app.log')  # Save to file
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Sri Lanka Tourism Platform API",
    description="API for tourist and service provider management with real-time chat",
    version="1.0.0"
)

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
app.include_router(chat_route, prefix="/chat")

app.include_router(event_route, prefix="/event")
app.include_router(service_providers_route, prefix="/service_provider")
app.include_router(service_provider_profile_route, prefix="/service_provider")
app.include_router(review_route, prefix="/review")
app.include_router(popular_toprated_route, prefix="/review")

@app.get("/")
def root():
    return {"message": "Firestore backend is running"}


if __name__ == "__main__" :
    uvicorn.run(app, host = "0.0.0.0", port = 9090, log_level = "info")