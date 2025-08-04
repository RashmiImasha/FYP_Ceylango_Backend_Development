from fastapi import FastAPI
from app.routes.category_route import router as category_route
from app.routes.destination_route import router as destination_route

# create Fast API app
app = FastAPI()

# register the routes
app.include_router(category_route, prefix="/category")
app.include_router(destination_route, prefix="/destination")

@app.get("/")
def root():
    return {"message": "Firestore backend is running"}