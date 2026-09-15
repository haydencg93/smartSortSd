# Smart Recycling Assistant

## Overview

The **Smart Recycling Assistant** is an AI-powered recycling guidance system designed to help users correctly identify whether an item should be placed in the trash or recycling.

The system uses a combination of **camera-based item recognition, AI classification, proximity sensors, and location-aware recycling information**. When a user approaches the device, proximity sensors activate the camera and AI recording logic. The system identifies the item and provides simple instructions explaining how it should be disposed of according to the recycling rules for the user's location.

The primary goal of the project is to **reduce recycling contamination, improve recycling-stream purity, and educate users about proper waste disposal**.

The initial prototype is designed around **Iowa City**, while allowing the system to provide location-specific recycling information so that the concept can eventually be adapted to other communities.

---

## Key Features

* 📷 **AI-powered item recognition**

  * Uses a camera to identify recyclable and non-recyclable objects.
  * Classification results can include accuracy, precision, recall, F1 score, and confusion-matrix analysis.

* 📡 **Proximity-based privacy**

  * Proximity sensors determine when a user is within **24 inches** of the system.
  * The camera and AI recording logic remain inactive when no user is nearby.

* ♻️ **Recycling guidance**

  * Provides clear instructions indicating whether an item belongs in recycling or trash.
  * Explains **why** an item belongs in a particular waste stream.

* 📍 **Location-aware recycling rules**

  * The system checks the user's location.
  * Recycling recommendations can be adjusted according to local waste-management rules.

* 🏷️ **Barcode identification**

  * GS1-compatible UPC/EAN barcodes can be used to identify retail products.

* ♿ **Accessible interface**

  * UI design follows WCAG 2.2 principles.
  * Uses simple text, clear visual indicators, arrows, and appropriate color contrast.

* 💰 **Low-cost prototype**

  * The total manufacturing cost of the prototype is designed to remain below **$300.00**.

---

# System Requirements

## Functional Requirements

### C1.4 — Privacy

The system shall use proximity sensors to ensure that the **camera and AI recording logic are only active when a user is within 24 inches** of the device.

This requirement is intended to minimize unnecessary data collection and prevent the camera from continuously recording when nobody is interacting with the system.

### C1.5 — Cost

The total manufacturing cost of the prototype shall **not exceed $300.00**.

The project prioritizes affordable, readily available hardware while maintaining sufficient processing capability for computer vision and AI classification.

---

# Standards

The project follows or references the following standards:

### GS1 Barcode Standards

**GS1 barcode standards** are used for identifying retail products through UPC/EAN barcode information.

Barcode identification can supplement AI-based image classification by providing an additional method of identifying products.

### WCAG 2.2

**Web Content Accessibility Guidelines (WCAG) 2.2** are used to guide the user-interface design.

The interface prioritizes:

* Readable text
* Sufficient color contrast
* Clear visual indicators
* Simple navigation
* Accessible symbols
* Easy-to-understand instructions

### FCC Part 15

**FCC Part 15** is considered when designing the electronic system.

The prototype will be designed to minimize unintended radio-frequency emissions and reduce potential interference with other electronic devices.

### ISO/IEC TS 4213:2022

This standard guides the **performance assessment of machine-learning classification models**.

The project can evaluate the AI classifier using metrics including:

* Accuracy
* Precision
* Recall
* F1 score
* Confusion matrix

These metrics help determine how reliably the system identifies different types of waste.

### ISO/IEC TS 12791:2024

This standard is used as guidance for evaluating **bias and fairness in classification systems**.

The project considers whether classification performance differs between different categories of objects, materials, packaging types, or other relevant groups.

### ISO/IEC TS 42119-2:2025

This standard provides guidance for **risk-based testing of AI systems**.

The project uses this concept to identify and test potential risks associated with incorrect AI classifications, including situations where an item is incorrectly identified as recyclable or non-recyclable.

---

# Societal Factors

## Public Health, Safety, and Welfare

The Smart Recycling Assistant is intended to improve public welfare by helping communities reduce waste contamination and increase the amount of material that can successfully enter recycling streams.

Improved recycling practices can reduce landfill reliance and contribute to reducing methane emissions associated with organic waste in landfills.

Safety considerations include:

* Low-voltage electronics
* Minimizing unintended radio-frequency emissions
* Environmental placement warnings
* Proximity-based camera activation
* Clear disposal instructions

---

## Social and Cultural Factors

Waste-disposal practices can vary between communities and cultures. The system addresses this issue by providing simple and understandable explanations for disposal decisions.

The UI uses:

* Universal symbols
* Arrows
* Simple text
* Clear visual feedback

These design choices can help reduce language barriers and make the system easier to understand.

Rather than simply telling users that an item is "trash" or "recycling," the system can explain **why** the item belongs in a particular waste stream. This provides an educational component that can help users develop better recycling habits.

---

## Environmental Factors

The project directly supports environmental sustainability by attempting to **increase the purity of recycling streams**.

Reducing contamination can increase the amount of recyclable material that can successfully be processed rather than being rejected and sent to a landfill.

Potential environmental benefits include:

* Reduced landfill reliance
* Reduced recycling contamination
* Increased recycling efficiency
* Improved recycling-stream purity
* Reduced waste-management impacts

---

## Economic Factors

Recycling contamination creates additional costs for municipalities and waste-management organizations.

By helping users correctly sort materials, the system aims to reduce contamination and potentially lower waste-management expenses, including **tipping fees** associated with contaminated or rejected material.

The project's **$300 prototype cost requirement** also encourages the development of an affordable solution that could potentially be scaled or adapted for broader use.

---

## Global Factors

Waste management and recycling contamination are global problems.

Although the prototype is initially designed and tested with **Iowa City recycling rules**, the system is intended to support location-specific recommendations.

The system can check its location and use the applicable recycling rules for that area. This allows the concept to be adapted to communities with different recycling requirements rather than assuming that every location follows the same disposal rules.

---

# Privacy

Privacy is a core design consideration of the system.

The camera and AI recording logic are controlled by proximity sensors. The system is designed so that these components are only active when a user is within **24 inches**.

This approach minimizes unnecessary camera operation and reduces the possibility of collecting visual information when the system is not actively being used.

---

# AI Classification

The AI component is responsible for identifying objects presented to the system.

A typical classification process is:

```text
User approaches device
        ↓
Proximity sensor detects user
        ↓
User is within 24 inches
        ↓
Camera activates
        ↓
Image captured
        ↓
AI classification
        ↓
Item identified
        ↓
Location-specific recycling rules checked
        ↓
Disposal recommendation
        ↓
User receives explanation
```

The AI model will be evaluated using appropriate classification metrics, including:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Testing will also consider potential classification risks and sources of bias.

---

# Project Goals

The primary goals of the Smart Recycling Assistant are to:

1. **Improve recycling accuracy**
2. **Reduce contamination in recycling streams**
3. **Educate users about proper waste disposal**
4. **Protect user privacy**
5. **Provide location-specific recycling information**
6. **Maintain an accessible and easy-to-use interface**
7. **Keep the prototype cost below $300**
8. **Evaluate AI performance using established classification metrics**
9. **Consider fairness and risk when deploying AI classification**
10. **Create a prototype that could eventually be adapted to other communities**

---

# Future Development

Potential future improvements include:

* Expanding the AI training dataset
* Supporting additional recyclable material categories
* Improving classification accuracy
* Adding more barcode-supported products
* Expanding location support beyond Iowa City
* Adding multilingual UI support
* Improving accessibility features
* Developing a mobile or web companion application
* Adding recycling-bin fill-level monitoring
* Collecting anonymized system performance statistics
* Improving AI risk and bias testing

---

# Conclusion

The Smart Recycling Assistant combines **AI, computer vision, proximity sensing, barcode identification, and location-aware recycling information** to address the problem of incorrect waste disposal.

By combining accurate classification with understandable explanations, the system is intended to do more than simply identify waste. It aims to **educate users, protect privacy, reduce recycling contamination, and encourage more sustainable waste-management practices** while remaining affordable enough to develop as a prototype.
