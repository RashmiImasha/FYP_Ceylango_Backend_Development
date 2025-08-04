from app.config.settings import settings
import json
import requests

def analyze_image_withAI(base64_image_str: str, prompt:str):
    api_key = settings.GOOGLE_API_KEY

    if not api_key:
        raise ValueError("Google API key is not set in the environment variables.") 
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    
    # prompt = (
    #     "Based on the image, identify the following:\n"
    #     "- The Location : \n"
    #     "- The district :\n"       
    #     "- Description including historical and cultural value."
    # )

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": base64_image_str
                    }
                }
            ]
        }]
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  
        result = response.json()
        return result
    
    except requests.exceptions.HTTPError as http_err:
        print("Status code:", response.status_code)
        print("Response body:", response.text)  # Print actual Gemini error
        raise
    
    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        return None
        
        
        


    
