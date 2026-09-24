Driver Safety and Assistance System

Project Overview

The Adeeb Smart Driver Safety and Assistance System is a computer-vision-based project designed to assist drivers, particularly during long-distance driving. The system uses computer vision and artificial intelligence (AI) to monitor driver behavior and provide alerts when signs of drowsiness, sleepiness, or distraction are detected.

The project was developed in Python and presented as part of a 2026 hackathon in the Gaza Strip. It was developed by Adeeb Abu Kmail.

The system monitors several driver-related indicators, including:

- Eye closure and eye activity.
- Gaze direction.
- Head orientation.
- Mobile-phone usage.
- General driver attention.

The system also supports integration with Arduino to provide an external and louder alarm, as well as support for ESP32-based hardware configurations and communication with a mobile phone over a local network.

For example, if the system detects that the driver's eyes have remained closed for approximately 2.5 seconds, it classifies the situation as a dangerous state and activates an audible alarm intended to alert the driver.

The system also supports direct voice interaction with a locally running Ollama-based AI model, which can be used to provide driving-related guidance and general voice interaction.

According to international road-safety literature, human-related factors constitute a major component of road-traffic risk. The project documentation references the following official sources:

- World Health Organization (WHO):
  https://www.who.int/ar/news-room/fact-sheets/detail/road-traffic-injuries

- WHO Global Status Report on Road Safety 2023:
  https://www.who.int/teams/social-determinants-of-health/safety-and-mobility/global-status-report-on-road-safety-2023

- U.S. National Highway Traffic Safety Administration (NHTSA):
  https://www.nhtsa.gov/road-safety/pedestrian-safety

- WHO Eastern Mediterranean Health Journal:
  https://www.emro.who.int/emhj-volume-32-2026/volume-32-number-7/road-traffic-injuries-and-road-safety-efforts-in-saudi-arabia.html

«Note: The sources above should be reviewed individually before claiming that more than 90% of crashes are specifically caused by drowsiness or distraction. The broader statement that human factors contribute substantially to road-traffic crashes is more scientifically defensible, while the exact percentage depends on the definition, dataset, country, and methodology used.»

---

AI Monitoring Modes

When the project is launched, the user can select from several AI-based monitoring configurations.

1. Laptop Camera

The first option uses the computer's built-in camera and is primarily intended for testing, demonstration, and development.

The system continuously analyzes the video stream to monitor the driver's face, eyes, gaze direction, and head orientation.

Drowsiness Detection

The system can be tested by closing the eyes for approximately 2.5 seconds. If prolonged eye closure is detected for this duration, the computer generates an audible alarm.

The system does not treat every eye closure as a dangerous event. Instead, it combines an eye-state threshold with a temporal condition, allowing normal short-duration blinks to be distinguished from prolonged eye closure.

Driver Attention Monitoring

The AI system analyzes the driver's:

- Gaze direction.
- Head orientation.
- Eye state.
- Potential mobile-phone usage.

A voice alert can be generated when the driver appears to be looking away from the forward direction for a sustained period or when mobile-phone usage is detected.

Mobile-Phone Detection

Mobile-phone usage can be detected using either:

1. A custom-trained YOLO object-detection model, or
2. An alternative rule-based detection approach that does not require a custom YOLO model.

Ollama AI Assistant

The system also provides voice interaction with a locally running AI assistant through Ollama.

The required model can be obtained using:

ollama pull llama3.2:3b

The local AI model can function as a driving-assistance assistant and can also be used for general voice interaction and entertainment.

---

2. Mobile Phone Camera

The second option allows a mobile phone to operate as the monitoring camera.

When this mode is activated, the computer acts as a local server and receives a live video stream from the mobile phone.

The average streaming rate is approximately 10 frames per second (FPS), although the actual frame rate depends on the computer, network conditions, camera configuration, and processing capabilities of the devices involved.

The received video is processed in real time using computer vision and AI techniques.

The system can generate alerts on both the mobile phone and the computer.

This configuration also supports:

- Voice communication between the phone and computer.
- Interaction with the AI assistant from both devices.
- Driver-attention monitoring.
- Gaze analysis.
- Head-orientation analysis.
- Mobile-phone detection.

Local Network Configuration

After starting the server, the required browser configuration can be enabled in Google Chrome by opening:

chrome://flags/#unsafely-treat-insecure-origin-as-secure

Then locate:

Insecure origins treated as secure

and enter the local IP address of the computer together with port 5000, for example:

http://10.10.10.216:5000

After completing the configuration, the mobile-phone monitoring mode can be tested over the local network.

---

3. Laptop Camera with Arduino and External Alarm

The third configuration combines the laptop camera with an Arduino board and an external alarm.

In this configuration, the computer performs the computer-vision and AI processing using the laptop camera, while the Arduino controls an external buzzer or alarm.

The purpose of this configuration is to provide a louder physical alert when a dangerous driver state is detected.

The Arduino therefore acts as an external hardware interface for the alert system.

---

4. ESP-Based Monitoring System

The fourth configuration provides support for monitoring systems based on ESP boards, including ESP32-based configurations.

The project is designed to allow different ESP-based hardware configurations depending on the required implementation.

For this configuration, the project's processing code should be installed and executed on a Raspberry Pi, which can act as the main processing and communication unit.

The ESP board can then be used as part of the hardware and alert infrastructure according to the selected system architecture.

---

Required Python Libraries

The required Python packages can be installed using:

pip install opencv-python numpy pillow mediapipe ultralytics pyttsx3 pyserial flask flask-socketio requests ollama SpeechRecognition sounddevice

---

Computer Vision and AI Techniques

Eye Closure Detection Using EAR

The system uses the Eye Aspect Ratio (EAR) to estimate whether the driver's eye is open or closed.

EAR is calculated using distances between specific facial landmarks around the eye. In simplified form, it represents the ratio between the vertical eye-opening distances and the horizontal eye width.

A higher EAR value generally corresponds to a more open eye, while a lower value generally corresponds to a partially or fully closed eye.

The project uses an EAR threshold of approximately:

EAR threshold = 0.205

However, the threshold alone is not sufficient to determine whether the driver is in danger. A temporal condition is also applied.

This is important because a normal blink may temporarily reduce the EAR below the threshold. Therefore, the system measures how long the eye remains below the threshold before classifying the condition as a dangerous event.

---

Driver-State Classification

The system uses several states to represent the driver's condition:

"NO FACE"

This state occurs when the system cannot detect a face in the camera frame.

"NORMAL"

The system considers the driver to be in a normal state when a face is detected and the eye-state measurements indicate that the eyes are sufficiently open.

"WARNING"

This state occurs when prolonged eye closure is detected for approximately 1.5 seconds, but the duration has not yet reached the critical alarm threshold.

No main audible danger alarm is triggered at this stage.

"DANGER"

This state occurs when the driver's eyes remain closed for approximately 2.5 seconds.

At this point, the system activates an audible alarm intended to alert the driver.

"PHONE DANGER"

This state occurs when the system detects potential mobile-phone usage while driving.

An audible warning is generated to alert the driver to the detected distraction.

---

Head-Orientation Estimation

Head orientation is estimated using facial landmarks.

The system obtains facial landmark coordinates and uses them to estimate the orientation of the driver's head relative to the camera.

In practical computer-vision implementations, this can be formulated as a 3D head-pose estimation problem, in which selected facial landmarks are compared with corresponding reference points from a predefined facial model.

The resulting geometric relationship can be used to estimate head orientation, commonly represented using angles such as:

- Yaw — left/right rotation.
- Pitch — up/down rotation.
- Roll — head tilt.

The system uses this information to determine whether the driver is significantly turning their head away from the forward direction.

---

Gaze Estimation

The system also analyzes the driver's gaze by tracking the position of the iris relative to the surrounding eye landmarks.

The iris position is evaluated continuously, and geometric calculations are used to estimate the direction in which the driver is looking.

This information allows the system to identify significant deviations from the expected forward gaze direction and generate an appropriate warning when the deviation persists.

---

Frame Smoothing

The system also applies temporal smoothing to reduce instability and sudden fluctuations in the detected measurements.

Instead of relying exclusively on a single video frame, measurements from multiple consecutive frames can be combined or averaged.

This helps reduce false detections caused by:

- Small facial movements.
- Camera noise.
- Temporary landmark inaccuracies.
- Minor changes between consecutive frames.

The result is a more stable monitoring system and more consistent alert behavior.

---

Project Technologies

The project combines several technologies and frameworks, including:

- Python
- OpenCV
- MediaPipe
- YOLO / Ultralytics
- NumPy
- Flask
- Flask-SocketIO
- Arduino
- ESP32
- Raspberry Pi
- Ollama
- Computer Vision
- Artificial Intelligence
- Real-Time Video Processing
- Speech Recognition
- Text-to-Speech

---

Project Author

Adeeb Abu Kmail