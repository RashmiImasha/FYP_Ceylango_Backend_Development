from fastapi import FastAPI
from app.routes.category_route import router as category_route

from app.routes.auth_route import router as auth_route

from app.routes.destination_route import router as destination_route

from app.routes.image_route import router as image_route


# create Fast API app
app = FastAPI()

# register the routes
app.include_router(category_route, prefix="/category")

app.include_router(auth_route, prefix="/auth")

app.include_router(destination_route, prefix="/destination")

app.include_router(image_route, prefix="/image")


@app.get("/")
def root():
    return {"message": "Firestore backend is running"}