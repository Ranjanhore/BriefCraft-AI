from jose import jwt

token = "PASTE_REAL_TOKEN_HERE"
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
print(decoded)
