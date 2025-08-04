from fastapi import FastAPI
from app.routes.category_route import router as category_route
from app.routes.auth_route import router as auth_route
# create Fast API app
app = FastAPI()

# register the routes
app.include_router(category_route, prefix="/category")
app.include_router(auth_route, prefix="/auth")

@app.get("/")
def root():
    return {"message": "Firestore backend is running"}