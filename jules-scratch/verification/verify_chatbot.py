from playwright.sync_api import sync_playwright, expect
import time

def run_verification():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Set a longer default timeout for all actions to handle slow loading
        page.set_default_timeout(60000)  # 60 seconds

        # 1. Arrange: Go to the Streamlit app and wait for it to be idle.
        page.goto("http://localhost:8501", wait_until="networkidle")

        # 2. Wait for a key element to be visible before proceeding.
        # This ensures the main application has loaded.
        expect(page.get_by_role("heading", name="🌱 Plant Disease Detector & Care Assistant")).to_be_visible()

        # 3. Act: Navigate to the Chatbot tab.
        page.get_by_role("button", name="💬 Chatbot").click()

        # 4. Assert: Check for the API key warning.
        expect(page.get_by_text("Please enter your OpenAI API key in the sidebar to use the chatbot.")).to_be_visible()

        # 5. Act: Enter a dummy API key in the sidebar.
        page.get_by_label("Enter your OpenAI API Key:").fill("DUMMY_KEY_FOR_TESTING")

        # Increased delay to ensure Streamlit registers the key.
        time.sleep(2)

        # 6. Act: Send a message in the chatbot.
        page.get_by_label("Ask your plant question here:").fill("Hello, what's wrong with my plant?")
        page.get_by_role("button", name="Send").click()

        # 7. Assert: Wait for the bot's error response (since the key is fake).
        expect(page.get_by_text("Sorry, an error occurred. Please check your API key and try again.")).to_be_visible()

        # 8. Screenshot: Capture the final result for visual verification.
        page.screenshot(path="jules-scratch/verification/verification.png")

        browser.close()

if __name__ == "__main__":
    run_verification()