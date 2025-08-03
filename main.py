from fastapi import FastAPI
from app.routes.category_route import router as category_route

# create Fast API app
app = FastAPI()

# register the routes
app.include_router(category_route, prefix="/category")

@app.get("/")
def root():
    return {"message": "Firestore backend is running"}