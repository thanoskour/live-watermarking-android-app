
# Development of an Application on Smart Devices for the Creation of Watermarked Multimedia Content

## Author
**Athanasios Koureas**

## Supervisor
Stavros D. Nikolopoulos

## University of Ioannina  
Department of Computer Science & Engineering  
July 2024  

## Abstract
This thesis explores the creation and implementation of a real-time image watermarking system using an Android camera application. The main goal is to embed user-specific data, such as a personal PIN and the device-specific Android ID, into digital images using a robust watermarking algorithm. By utilizing the CameraX library and integrating Python scripts through the Chaquopy plugin, we developed a system that ensures the security and authenticity of digital images captured in real-time.

Extensive testing has shown that this system effectively protects digital images from unauthorized use and alterations. The watermarking method is highly resistant to common attacks like cropping and compression, although the system's performance may vary based on the device's processing power.

Future research could focus on optimizing the algorithm, expanding its use to other media types like video and audio.

## Table of Contents
1. [Introduction](#introduction)
2. [Theoretical Background](#theoretical-background)
3. [The Model](#the-model)
4. [System Architecture and Deployment](#system-architecture-and-deployment)
5. [Evaluation](#evaluation)
6. [Conclusion](#conclusion)
7. [Installation](#installation)
8. [Technologies Used](#technologies-used)
9. [How to Use](#how-to-use)
10. [License](#license)
11. [Contributing](#contributing)

## Introduction
In this thesis, we introduce the essential techniques for embedding hidden information in digital images, with a focus on digital watermarking using an Android camera. The objectives are to create a robust real-time watermarking system that ensures the security and authenticity of digital images.

## Theoretical Background
The thesis explores various techniques for hiding information in digital images, such as encoding integers as Self-inverting Permutations (SiPs) and embedding them using their Two-Dimensional Matrix (2DM) representation.

## The Model
A detailed explanation of how user and device-specific data are combined to generate a unique watermark is provided, along with the process of embedding the watermark into the image using a camera application.

## System Architecture and Deployment
This section discusses the modular architecture of the system, including the watermarking algorithm, Android app integration using the CameraX library, and Python scripting through Chaquopy.

## Evaluation
Performance metrics such as Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) were used to evaluate the quality of watermarked images. The system proved effective in embedding secure watermarks without compromising image quality.

## Conclusion
The thesis concludes that the real-time watermarking system effectively secures digital images from unauthorized use, with future research potential in optimizing the algorithm and extending it to other media types.

## Installation
To set up and run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/username/repository-name.git
   cd repository-name
   ```

2. Open the project in Android Studio.

3. Install the necessary dependencies (e.g., CameraX, Chaquopy) as defined in the `build.gradle` file.

4. Connect an Android device or run the emulator.

5. Run the application.

## Technologies Used
- Android Studio
- CameraX library
- Python via Chaquopy plugin
- OpenCV for image processing
- NumPy and SciPy for numerical computations
- Java/Kotlin for Android development

## How to Use
1. Launch the application.
2. Capture an image using the app.
3. Enter your PIN when prompted.
4. The app will embed a watermark into the image based on your input and device-specific information.
5. View and access the watermarked images in the specified storage directory.

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with any improvements or fixes.

## Full Thesis
For more detailed information, you can download the full thesis [here](https://github.com/thanoskour/live-watermarking-android-app/blob/master/FinalReport_4392.pdf).

## Keywords
- Image Watermarking
- Android Camera Application
- Watermarking Algorithm
- Personal PIN
- Android ID
- Image Processing
- Data Embedding
- Digital Authenticity
- Live Watermarking
