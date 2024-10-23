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

## Full Thesis
For more detailed information, you can download the full thesis [here](path/to/your/thesis.pdf).

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
