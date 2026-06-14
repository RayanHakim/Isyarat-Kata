IsyaratKata: AI-Powered Sign Language Keyboard
IsyaratKata is a high-performance, real-time sign language recognition virtual keyboard. Built without the need for heavy Deep Learning model training, this project utilizes a custom Multi-Angle Geometric Finger State Algorithm combined with Spatial Face Tracking to recognize over 100 sign language vocabularies instantly using just a standard webcam.

This project was specifically engineered to bypass global TensorFlow/Protobuf dependency conflicts on Windows by injecting custom module mocks and relying purely on MediaPipe's lightweight geometric landmarks.

✨ Key Features
💯 100+ Built-in Vocabularies: Instantly recognizes Alphabets (A-Z), Numbers (0-10), Daily Actions, Family terms, and Time/Adjectives.

✋ Multi-Hand Synchronization: Seamlessly tracks both Left and Right hands simultaneously to detect 2-handed signs like "Berdoa", "Menyerah", and "Bertinju".

🧑‍🦲 Spatial Face Interaction: Integrates face detection to map hand coordinates relative to facial landmarks. (e.g., pointing near the ear outputs "Mendengar", while pointing near the eye outputs "Melihat").

⚡ Zero-Training Architecture: Uses pure mathematical Euclidean distance and joint-angle comparisons (MCP, PIP, TIP) instead of heavy neural networks, resulting in maximum FPS even on CPU.

🖥️ Interactive HUD Overlay: Built-in scrollable Help Menu (h), real-time text compiler, and a 1-click Export to .txt Notepad feature (e).

🛠️ Tech Stack
Language: Python 3.12 (Strictly recommended for MediaPipe stability)

Computer Vision: OpenCV (cv2)

Landmark Tracking: Google MediaPipe (Hands & Face Detection)

Matrix Operations: NumPy

🚀 Installation & Setup
Clone the Repository

Bash
git clone https://github.com/RayanHakim/isyarat-kata.git
cd isyarat-kata
Install Dependencies
(Note: It is recommended to run this on a clean Python 3.12 environment)

Bash
pip install opencv-python mediapipe numpy
Run the Application

Bash
python app_final.py
🎮 How to Use & Key Bindings
Once the camera window opens, position yourself clearly in the frame. The system will automatically detect your face and hands. Hold a specific hand gesture for 1.0 second to type the word into the console.

Keyboard Controls:

h : Toggle (Hide/Show) the vocabulary help menu overlay.

w : Scroll the help menu UP.

s : Scroll the help menu DOWN.

c : Clear the currently typed sentence from the screen.

e : Export the typed sentence to a .txt file automatically.

q : Quit the application.

🧠 Core Logic Explanation
Instead of using traditional CNN/LSTM models, this application uses a Deterministic Geometric State Array.

The system extracts 21 3D landmarks from each detected hand.

It calculates the Euclidean distance between the fingertips (TIP) and the lower joints (PIP/MCP) to determine if a finger is extended 1 or curled 0.

The resulting array (e.g., [0, 1, 1, 0, 0] for the "Peace/V" sign) is cross-referenced with spatial data (e.g., Is the hand near the nose? Is the thumb horizontal?) to eliminate key-collisions.

To prevent protobuf and tensorflow import crashes common in newer MediaPipe builds, the script implements unittest.mock.MagicMock() at runtime.
