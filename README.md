# Driver-safety-and-assistance-system
A project designed to assist long-distance drivers; it utilizes computer vision and AI to alert the driver if they doze off or fall asleep while driving.
Adeeb Smart Driver Monitoring and Hazard Alert System

The Adeeb Smart Project is an intelligent driver-monitoring and hazard-alert system designed to detect potential risks during driving, such as drowsiness or loss of attention. The project was developed using Python and relies on computer vision and artificial intelligence technologies.

The system also supports integration with Arduino to add a louder external alarm, as well as support for ESP boards of different types.

When the project is launched, the user is provided with several AI monitoring options:

1. Laptop Camera

The first option is the laptop's built-in camera, which is primarily intended for testing and demonstration purposes. Through this option, the user can test the system by closing their eyes for a period of up to five seconds. If the system detects prolonged eye closure, an alarm sound will be triggered from the laptop.

The AI system also analyzes the driver's gaze direction and head orientation. It is programmed to provide a voice alert when the driver is not looking forward or when the driver is using a mobile phone.

Mobile-phone usage can be detected automatically, either through a custom-trained YOLO model or through an alternative detection method that does not require custom YOLO training.

The system also provides voice interaction with an AI assistant powered by Ollama, specifically using:

ollama pull llama3.2:3b

The Ollama model can respond as a driving guidance assistant and can also be used for entertainment and general interaction.

2. Mobile Phone Camera

The second option allows the user to use a mobile phone as the monitoring camera. When this option is activated, the computer operates as a server and receives a live video stream from the mobile phone at approximately 10 frames per second.

The received video is analyzed in real time using artificial intelligence and computer vision. The system can provide alerts on both the mobile phone and the computer.

This option also supports voice communication between the mobile phone and the computer, as well as interaction with the AI assistant from both devices. The same AI functions available in the laptop-camera mode are maintained, including driver attention monitoring, gaze analysis, head-orientation analysis, and mobile-phone detection.

To enable the required browser functionality on the computer, after starting the server, open Google Chrome and enter the following address in the browser:

chrome://flags/#unsafely-treat-insecure-origin-as-secure

Then locate:

Insecure origins treated as secure

Enter the computer's local IP address together with port 5000, for example:

http://10.10.10.216:5000

After completing these steps, the mobile-phone monitoring mode can be tested.

3. Laptop Camera with Arduino and External Alarm

The third option combines the laptop camera with an Arduino board and an external alarm. This configuration allows the AI monitoring system to use the laptop camera for computer-vision analysis while the Arduino controls an external buzzer or alarm with a higher sound level.

4. ESP-Based Monitoring System

The fourth option provides support for an ESP-based monitoring system. The project is intended to be compatible with ESP boards of different types, depending on the required hardware configuration.

For the system to operate correctly in this configuration, the code should be installed and executed on a Raspberry Pi.

Project Author:
Adeeb Abu Kmail
