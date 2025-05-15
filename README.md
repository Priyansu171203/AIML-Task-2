# TASK-2 SPEECH RECOGNITION SYSTEM
COMPANY: CODTECH IT SOLUTIONS

NAME: Priyanshu V. patel

INTERN ID: C0DF191

DOMAIN: Artificial Intelligence Markup Language.

DURATION: 4 WEEEKS

MENTOR: NEELA SANTOSH

# Speech Recognition System
This Python script is a real-time speech-to-text converter that captures audio through a microphone, transcribes it using Google’s Web Speech API, and saves raw audio as timestamped WAV files. Developed with the speech_recognition library, it offers an easy-to-use tool for voice-based transcription. The script is designed to run continuously, capturing and transcribing speech until manually stopped, making it ideal for use cases like interviews, lectures, or dictation.

Core Features and Functionality The main function of this script is to transcribe spoken words into text. It uses the speech_recognition library, which interfaces with Google’s Web Speech API. As the microphone captures audio, the script transcribes it in near real-time, adjusting for ambient noise to ensure higher accuracy in transcription. The ability to save audio files in the WAV format, each timestamped, adds value by providing users with a record of both the transcribed text and the raw audio for future reference or analysis.

Real-Time Processing and Infinite Loop Design The script is designed to work in an infinite loop, continuously transcribing until the user manually stops it. This feature is particularly useful for long sessions where a user needs to record an entire lecture or interview. The script automatically processes and updates transcriptions in real-time, ensuring a seamless experience for users who need immediate feedback from their recordings.

Target Audience and Use Cases The script is valuable for a range of professionals. Students and researchers can use it to transcribe lectures, meetings, or discussions. Journalists can utilize it for transcribing interviews, saving time and effort compared to manual transcription. Developers can also use the script as the foundation for creating voice-driven applications, such as voice assistants or chatbots. Additionally, the script can be extended to work with other speech-to-text APIs, such as Whisper or Azure Speech Services, for better flexibility and customization.

Practical Applications and Extensibility The modular design of the script allows easy customization and extension. Developers can add features like keyword detection, enabling the script to listen for specific words or phrases. This would allow it to trigger actions based on certain keywords, such as starting a process or sending a command to another system. Furthermore, the raw audio files can be uploaded to cloud storage solutions, like AWS S3 or Google Cloud Storage, making it easier to store and manage large amounts of data.

Additionally, integrating a graphical user interface (GUI) could enhance user experience. A GUI could allow users to easily start or stop the transcription, view the real-time text output, and interact with audio files more intuitively. This would make the tool more accessible, especially for non-technical users.

Tools and Resources The project was developed using Visual Studio Code (VS Code), which provided features such as code linting, debugging, and version control via Git. These tools facilitated efficient development and allowed for easy management of the codebase. ChatGPT played a crucial role in helping the developer overcome challenges like ambient noise calibration and exception handling, as well as offering suggestions for improving code structure and performance.

YouTube tutorials were invaluable in the early stages, providing hands-on guidance on how to set up the microphone, work with the speech_recognition library, and troubleshoot common issues. Websites like GeeksforGeeks were also helpful in understanding Python’s file handling and exception management. Online communities like r/learnpython and r/Python on Reddit further provided solutions to specific issues, such as microphone index selection and handling edge cases like interruptions in speech.

Conclusion This real-time speech-to-text converter is a versatile tool that can be used in various fields, including research, journalism, and app development. It simplifies the process of transcribing speech into text, with the added benefit of saving raw audio files. The script’s modular design allows it to be extended with new features such as keyword detection, cloud storage integration, and even a GUI for easier use. With the help of online resources and tools like Visual Studio Code, this script offers a practical and customizable solution for voice-based transcription tasks

# OUTPUT
![Image](https://github.com/user-attachments/assets/bd309ffd-50d5-4e75-bbeb-4cbd3174a855)
