from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import os

# Setup
options = Options()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(service=Service(), options=options)
driver.set_page_load_timeout(60)

# Open site
url = "https://www.naukri.com/intership-jobs"
driver.get(url)

data = []
page = 1

# Define the base filename
base_filename = "naukri_all_internships.csv"
filename = base_filename

# Check if the file already exists and create a new filename if it does
counter = 1
while os.path.exists(filename):
    filename = f"{base_filename[:-4]}_{counter}.csv"  # Create a new filename with a counter
    counter += 1

try:
    while True:
        print(f"📄 Scraping page {page}...")
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "cust-job-tuple"))
            )

            cards = driver.find_elements(By.CLASS_NAME, "cust-job-tuple")
            if not cards:
                print("✅ No job cards found. Exiting...")
                break

            for card in cards:
                try:
                    title = card.find_element(By.CSS_SELECTOR, "h2 a.title").text.strip()
                    link = card.find_element(By.CSS_SELECTOR, "h2 a.title").get_attribute("href")
                except:
                    title, link = "", ""

                try:
                    company = card.find_element(By.CSS_SELECTOR, ".comp-name").text.strip()
                except:
                    company = ""

                try:
                    experience = card.find_element(By.CLASS_NAME, "expwdth").text.strip()
                except:
                    experience = ""

                try:
                    salary = card.find_element(By.CSS_SELECTOR, ".sal-wrap span[title]").text.strip()
                except:
                    salary = ""

                try:
                    location = card.find_element(By.CLASS_NAME, "locWdth").text.strip()
                except:
                    location = ""

                try:
                    skills = ", ".join([li.text for li in card.find_elements(By.CSS_SELECTOR, ".tags-gt li")])
                except:
                    skills = ""

                data.append({
                    "Title": title,
                    "Company": company,
                    "Experience": experience,
                    "Salary": salary,
                    "Location": location,
                    "Skills": skills,
                    "Link": link
                })

            # 🔁 Click next if available
            try:
                next_button = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, "//a[contains(@class,'styles_btn-secondary__2AsIP')]/span[text()='Next']/.."))
                )
                if next_button.is_enabled():
                    driver.execute_script("arguments[0].click();", next_button)
                    time.sleep(2)  # Wait for the next page to load
                    page += 1
                else:
                    print("✅ No more pages left.")
                    break
            except Exception as e:
                print("✅ No more pages left or error occurred: ", e)
                break

        except Exception as e:
            print(f"🚨 Error on page {page}: {e}")
            break

except KeyboardInterrupt:
    print("🛑 Stopped by user! Saving collected data...")

finally:
    # ✅ Always save data whether it finished or stopped
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"✅ Data saved! {len(data)} listings written to {filename}.")
    driver.quit()