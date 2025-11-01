from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import requests
import os
from PIL import Image
from io import BytesIO

# Setup Chrome with webdriver_manager
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

options = webdriver.ChromeOptions()
options.add_argument("--headless")   # run in background
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Step 1: Go to Google Images
search_query = "lehenga for girls"
driver.get("https://www.google.com/imghp?hl=en")

# Step 2: Enter search query
box = driver.find_element(By.NAME, "q")
box.send_keys(search_query)
box.send_keys(Keys.RETURN)

# Step 3: Scroll to load more images
for _ in range(100):  # scroll more times for more images
    driver.execute_script("window.scrollBy(0, document.body.scrollHeight)")
    time.sleep(3)

# Step 4: Collect image URLs
img_elements = driver.find_elements(By.CSS_SELECTOR, "img")
img_urls = []
for img in img_elements:
    src = img.get_attribute("src")
    if src and src.startswith("http"):
        img_urls.append(src)

print("Collected", len(img_urls), "image URLs")


# Step 5: Save images
os.makedirs("/Users/heerpatel/Desktop/fashion_project/data/lehenga_scraped", exist_ok=True)
pics = 0
for i, url in enumerate(img_urls):
    try:
        # if url.startswith("http"):
        #     img_data = requests.get(url).content
        #     with open(f"couture/{search_query}_{i}.jpg", "wb") as f:
        #         f.write(img_data)
        img_data = requests.get(url, timeout=10).content
        img = Image.open(BytesIO(img_data))
        if img.width < 150 or img.height < 150:
            continue  # skip tiny images/logos
        img.save(f"/Users/heerpatel/Desktop/fashion_project/data/lehenga_scraped/{search_query.replace(' ', '_')}_{i}.jpg")
        pics+=1
    except Exception as e:
        print("Could not save image:", e)

print ("Saved", pics, "images")
driver.quit()
