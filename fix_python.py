import os
import json

# 1. Automatically find your computer's correct user folder
# This works whether your user is 'prabh', 'prabhav', or 'prabhavkhare'
kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")

# 2. Create the hidden folder if it doesn't exist
os.makedirs(kaggle_dir, exist_ok=True)

# 3. Define the credentials (I took these from your screenshot)
data = {
    "username": "prabhavkhare",
    "key": "a7e8914b5554dbaf759f6baf50afe66b"
}

# 4. Write the file safely
file_path = os.path.join(kaggle_dir, "kaggle.json")
with open(file_path, "w") as f:
    json.dump(data, f)

print(f"✅ Success! Fixed file at: {file_path}")
print("You can now run 'kaggle datasets list'")