# realtime.py
import time
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configuration
READ_INTERVAL = 0.05  # seconds between reads (~20 Hz)
WINDOW_DURATION = 5   # seconds per classification window

# Launch browser
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # run headless if preferred
options.add_argument('--disable-gpu')
driver = webdriver.Chrome(options=options)


def read_realtime_data():
    driver.get("http://192.168.2.38/")

    # Wait for view selector
    wait = WebDriverWait(driver, 10)
    view_selector = wait.until(
        EC.presence_of_element_located((By.ID, "viewSelector"))
    )

    # Click "Simple" tab
    simple_tab = view_selector.find_elements(By.TAG_NAME, "li")[-1]
    simple_tab.click()

    x_vals, y_vals, z_vals = [], [], []
    start_time = time.time()

    while time.time() - start_time < WINDOW_DURATION:
        value_spans = driver.find_elements(By.CLASS_NAME, "valueNumber")
        if len(value_spans) >= 3:
            try:
                x_vals.append(float(value_spans[0].text))
                y_vals.append(float(value_spans[1].text))
                z_vals.append(float(value_spans[2].text))
            except ValueError:
                continue
        time.sleep(READ_INTERVAL)

    return np.array(x_vals), np.array(y_vals), np.array(z_vals)


def close_driver():
    driver.quit()