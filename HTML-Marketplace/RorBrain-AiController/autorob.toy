import pyautogui
import datetime
import os

def capture_full_screen(save_dir="screenshots"):
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Generate a timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = os.path.join(save_dir, f"screenshot_{timestamp}.png")

        # Capture the screen
        screenshot = pyautogui.screenshot()

        # Save the screenshot
        screenshot.save(file_path)

        print(f"✅ Screenshot saved at: {file_path}")
        return file_path

    except Exception as e:
        print(f"❌ Error capturing screenshot: {e}")
        return None

if __name__ == "__main__":
    capture_full_screen()
