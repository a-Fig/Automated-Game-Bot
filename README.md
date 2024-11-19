# Automated Albion Online Bot
## Overview
This project is a fully automated game bot developed for Albion Online, built using Python. The bot is designed to perform various in-game tasks autonomously by utilizing object detection, multi-threading, and image processing to scan the game environment, interact with objects, and perform actions such as farming, fighting, and peace mode activation.


The bot demonstrates advanced skills in computer vision, automation, and multi-threaded programming, and serves as an excellent showcase of my capabilities as a developer.

## Video Fotage

[www.youtube.com](https://youtu.be/_WJ6Z3qbqEo)
https://youtu.be/R9eL22sFTKY

## Features
#### Object Detection: 
Utilizes YOLOv11 for detecting various game objects such as mobs, farmable resources, and player characters.
#### Spell Casting:
Detects the status of spells and casts appropriate abilities, such as shields and attacks, based on in-game conditions.
#### Automated Interactions:
Automates actions like farming, fighting, and using items (e.g., shields, healing spells) based on object detection.
#### Obstacle Detection: 
Recognizes in-game obstacles and adjusts the movement path to avoid them.
#### Multithreading: 
Implements multi-threading to handle simultaneous scanning and action execution, ensuring efficient performance.

## Technologies Used
#### Python: 
The primary language used for developing the bot.
#### OpenCV: 
For image processing and object detection.
#### YOLOv11: 
For object detection and classification.
#### PyAutoGUI: 
For automating keyboard and mouse inputs.
#### Keyboard: 
For detecting hotkey inputs to trigger actions such as taking screenshots.
#### Multiprocessing:
For managing multiple threads to handle different tasks concurrently.
