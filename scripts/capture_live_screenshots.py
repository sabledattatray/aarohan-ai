"""Capture fresh, high-resolution live screenshots of Aarohan AI 2.0 Studio."""

import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from PIL import Image

DOCS_DIR = Path(r"D:\Datta Sable\AAROHAN\docs\assets\screenshots")
ROOT_DIR = Path(r"D:\Datta Sable\AAROHAN\assets\screenshots")
DOCS_DIR.mkdir(parents=True, exist_ok=True)
ROOT_DIR.mkdir(parents=True, exist_ok=True)

def save_both(im_path: Path, name: str):
    p1 = DOCS_DIR / name
    p2 = ROOT_DIR / name
    im = Image.open(im_path)
    if name.endswith(".jpg") or name.endswith(".jpeg"):
        im = im.convert("RGB")
        im.save(p1, "JPEG", quality=95)
        im.save(p2, "JPEG", quality=95)
    else:
        im.save(p1, "PNG")
        im.save(p2, "PNG")
    print(f"Saved {name} to docs and assets")

def main():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1500,980")
    options.add_argument("--disable-gpu")
    options.add_argument("--force-device-scale-factor=1.25")

    driver = webdriver.Edge(options=options)
    try:
        # 1. Clean Home Canvas
        print("Navigating to Aarohan Studio...")
        driver.get("http://127.0.0.1:8000/app")
        time.sleep(3)

        temp1 = DOCS_DIR / "temp1.png"
        driver.save_screenshot(str(temp1))
        save_both(temp1, "aarohan_dark_interface.png")

        # 2. Submit reasoning query
        print("Submitting rule reasoning query...")
        chat_input = driver.find_element(By.ID, "chat-input")
        chat_input.clear()
        chat_input.send_keys("What is the contact deadline for LEVEL-A accounts?")
        send_btn = driver.find_element(By.ID, "chat-send-btn")
        send_btn.click()
        
        # Wait for SSE stream to complete
        time.sleep(6)
        temp2 = DOCS_DIR / "temp2.png"
        driver.save_screenshot(str(temp2))
        save_both(temp2, "aarohan_reasoning_verification.jpg")

        # 3. Submit identity & web search query
        print("Submitting identity & live search query...")
        chat_input = driver.find_element(By.ID, "chat-input")
        chat_input.clear()
        chat_input.send_keys("Who created AAROHAN AI and what is your official website?")
        send_btn.click()
        time.sleep(6)

        temp3 = DOCS_DIR / "temp3.png"
        driver.save_screenshot(str(temp3))
        save_both(temp3, "aarohan_chat_studio.png")

        # 4. Switch to Cognitive Memory / Multimodal workspace
        print("Switching to Images workspace...")
        try:
            nav_images = driver.find_element(By.ID, "nav-images")
            nav_images.click()
            time.sleep(3)
            temp4 = DOCS_DIR / "temp4.png"
            driver.save_screenshot(str(temp4))
            save_both(temp4, "aarohan_multimodal_studio.jpg")
        except Exception as e:
            print("Workspace switch error:", e)

        print("\nAll fresh live screenshots captured successfully!")

    finally:
        driver.quit()
        # Clean up temp files
        for p in [temp1, temp2, temp3, DOCS_DIR / "temp4.png"]:
            if p.exists():
                p.unlink()

if __name__ == "__main__":
    main()
