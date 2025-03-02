# NHG Project - Automated Diabetic Foot Screening System
veNTUre Project
---

## **1. Introduction**
This project focuses on segmenting **thermal images of feet** to aid in diabetic foot screening. The objective is to accurately **isolate the foot** while eliminating background elements such as **noise, overlays, and text**.

Three segmentation approaches were tested:
- **Color-Based Segmentation in HSV**
- **Segment Anything Model (SAM)**
- **Greyscale Thresholding with Otsu’s Method**

Each method is described below, including the **process, advantages, and limitations**.

---

## **2. Methods Used**

### **Color-Based Segmentation in HSV**
#### **Process:**
- Convert the image to **HSV format** to enhance differentiation based on temperature variations.
- Define **HSV thresholds** to isolate the foot region and suppress background details.
- Generate a **binary mask** where values within the HSV range are retained.
- Apply **morphological closing** to remove small holes.
- Identify and retain the **largest connected component**.
- Apply the mask to the original image to extract the foot.

#### **Advantages:**
- Computationally efficient.
- Works well when the foot has a **distinct temperature profile**.

#### **Limitations:**
- Requires **manual tuning** of HSV values.
- **Ineffective** when the background has similar colors.
- Text overlays can interfere with segmentation.

---

### **Segment Anything Model (SAM)**
#### **Process:**
- Load the **SAM ViT-Huge model** and initialize **automatic mask generation**.
- Generate **segmentation masks** for different objects in the image.
- Merge all detected masks into a **single mask**.
- Retain the **largest connected component** (assuming it represents the foot).
- Apply **morphological closing** for refinement.
- Extract the foot while preserving **thermal data**.

#### **Advantages:**
- Adaptable to **varying image conditions** without predefined thresholds.
- Handles **complex backgrounds** better than simpler methods.

#### **Limitations:**
- May **over-segment** and include unwanted elements.
- Requires **significant computational power**.
- **Filtering needed** to remove overlays and irrelevant regions.

---

### **Greyscale Thresholding with Otsu’s Method**
#### **Process:**
- Convert the image to **grayscale**.
- Apply **Gaussian blur** to minimize noise.
- Use **Otsu’s method** to determine an **optimal threshold** for segmentation.
- Apply **morphological closing** to refine the mask.
- Identify and retain the **largest connected component**.
- Extract the foot region using the **final mask**.

#### **Advantages:**
- **Fast** and **fully automatic**.
- Works well for **images with clear foreground-background separation**.

#### **Limitations:**
- Struggles when the **foot and background have similar intensities**.
- Often **includes overlays and text**.
- Less effective in **varied lighting conditions**.

---

## **3. Future Suggestions**
- Implement **automated overlay removal** using **color-based filtering**.
- Explore **deep learning models trained on thermal images**.
- Utilize **bounding box or point-based segmentation with SAM** for improved precision.

---

## **4. What Have We Learned?**
### **1. OpenCV for Python Developers**
This course provided a strong foundation in **image processing** using OpenCV, covering key techniques for **object detection and feature extraction**.
- **Image Manipulation & Thresholding:** Understanding **pixel operations** and applying **simple/adaptive thresholding**.
- **Edge Detection & Contours:** Using **Canny edge detection** and **contour analysis** to extract object boundaries.
- **Object & Face Detection:** Applying **Haar cascades** for detecting facial features and **skin detection** for biometric applications.

### **2. Deep Learning: Image Recognition**
This course introduced **CNNs and AI-powered image recognition**, focusing on automated **feature extraction and classification**.
- **Understanding Deep Learning for Images:** Covering **image processing basics**, **CNN architectures**, and **hierarchical feature extraction**.
- **Image Recognition Fundamentals:** Learning about **data preprocessing, feeding structured data into deep networks**, and developing **custom image recognition systems**.
- **Success Metrics & Optimization:** Evaluating model **accuracy, precision, recall, and F1-score** to ensure reliable performance.

