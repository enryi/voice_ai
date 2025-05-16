from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

options = Options()
options.add_argument("--headless=new")  # puoi attivarlo quando tutto funziona
options.add_argument("--window-size=1920,1080")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")

driver = webdriver.Chrome(options=options)

try:
    driver.get("https://www.youtube.com/@NovaLectio/videos")
    
    wait = WebDriverWait(driver, 10)
    
    # Clicca sul bottone (es. accetta cookie) se presente
    try:
        cookie_button = wait.until(EC.element_to_be_clickable((
            By.XPATH,
            '/html/body/c-wiz/div/div/div/div[2]/div[1]/div[3]/div[1]/form[2]/div/div/button/span'
        )))
        cookie_button.click()
        print("✔ Bottone cliccato (es. cookie)")
        time.sleep(2)  # aspetta il completamento dell'animazione/modal
    except:
        print("ℹ Nessun bottone da cliccare (già accettato o layout diverso)")

    time.sleep(5)  # aspetta caricamento video

    # Scroll per caricare più contenuti
    last_height = driver.execute_script("return document.documentElement.scrollHeight")
    for _ in range(10):
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.documentElement.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    # Trova tutti i link dei video
    video_links = driver.find_elements(By.CSS_SELECTOR, 'a[href^="/watch?v="]')
    unique_links = set(link.get_attribute("href") for link in video_links if link.get_attribute("href"))

    print(f"Trovati {len(unique_links)} video:\n")
    with open("ai_learning/links.txt", "w", encoding="utf-8") as f:
        for link in unique_links:
            print(link)
            f.write(link + "\n")

finally:
    driver.quit()