from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.category_route import router as category_route
from app.routes.destination_route import router as destination_route

# Create FastAPI app
app = FastAPI()

# Setup CORS
origins = [
    "http://localhost:5173",  # Vite (React/Vue) dev server
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # You can use ["*"] for all origins (not recommended for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register the routes
app.include_router(category_route, prefix="/category")
app.include_router(destination_route, prefix="/destination")

@app.get("/")
def root():
    return {"message": "Firestore backend is running"}
