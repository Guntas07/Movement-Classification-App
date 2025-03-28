import time
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configuration
READ_INTERVAL = 0.05  # seconds between reads from the website
WINDOW_DURATION = 5   # seconds per classification window

# Launch browser, use headless, so it does not open a new window
options = webdriver.ChromeOptions()
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)


def read_realtime_data():
    # This is the server that PhyPhox outputs the data to
    # driver.get("http://192.168.2.38/") # Address of my house
    driver.get("http://10.216.24.230/") # Address at stauffer

    # Wait up to 10 seconds for view selector to load
    wait = WebDriverWait(driver, 10)
    view_selector = wait.until(
        EC.presence_of_element_located((By.ID, "viewSelector"))
    )

    # Click "Simple" tab, where we access our data from
    simple_tab = view_selector.find_elements(By.TAG_NAME, "li")[-1]
    simple_tab.click()

    # Store the values in an array and start timing
    x_vals, y_vals, z_vals = [], [], []
    start_time = time.time()

    # While we have not passed 5 seconds (length of a window), store the x,y, and z values
    while time.time() - start_time < WINDOW_DURATION:
        # Find the HTML elements with class = "valueNumber"
        # Value_spans[0] = x, [1] = y, [2] = z
        value_spans = driver.find_elements(By.CLASS_NAME, "valueNumber")
        if len(value_spans) >= 3:
            try:
                # Stores the acceleration values in their corresponding lists
                x_vals.append(float(value_spans[0].text))
                y_vals.append(float(value_spans[1].text))
                z_vals.append(float(value_spans[2].text))
            except ValueError:
                # If we can't find a value, continue and try again
                continue
        # Take a reading after delaying 0.05 seconds
        time.sleep(READ_INTERVAL)

    # Return the arrays as a numpy array
    return np.array(x_vals), np.array(y_vals), np.array(z_vals)


def close_driver():
    driver.quit()
