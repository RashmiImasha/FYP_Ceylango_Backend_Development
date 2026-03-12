# AI-Aided Content Management System for Tourism Industry

![Python](https://img.shields.io/badge/Python-3.x-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Flutter](https://img.shields.io/badge/Flutter-Mobile-blue)
![React](https://img.shields.io/badge/React-Web-61DAFB)
![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-purple)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![AWS](https://img.shields.io/badge/AWS-Cloud-orange)
![LLM](https://img.shields.io/badge/AI-LLM-red)

AI-powered tourism platform designed to enhance the travel experience in **Sri Lanka** using **Generative AI, vector search, and location-aware services**.

The system integrates **mobile applications, web portals, and AI backend services** to provide intelligent tourism assistance including **content generation, multilingual communication, and personalized travel planning**.

---

# Problem Statement

Sri Lanka currently lacks a locally developed tourism platform that effectively utilizes modern technologies such as **Artificial Intelligence**.

Existing travel platforms often fail to highlight:

- Hidden attractions  
- Nearby services (night shops, events, vehical rentals etc)
- Historical and cultural locations  

Additionally, tourists frequently face **language barriers** when communicating with locals and struggle to access reliable travel information.

This project addresses these challenges by building an **AI-powered tourism platform** that provides intelligent recommendations, multilingual assistance, and automated tourism content generation.

---

# Proposed Solution

The platform provides an **AI-driven tourism assistance system** that helps tourists discover destinations and services, learn about locations, and plan trips efficiently.

### Key Features

- Image-based tourism content generation
- Multilingual AI chatbot
- Intelligent trip planning
- Location-aware recommendations
- Nearby services and emergency contacts

---

# System Architecture

The system follows a **Full-Stack AI Architecture** integrating mobile apps, web portals, and AI services.

## Frontend

- Flutter mobile application for tourists
- React web portal for administrators
- React web portal for service providers

## Backend

- Python-based API services using FastAPI
- AI processing services for LLM pipelines

## Databases

- Firebase Firestore (Operational data)
- Pinecone Vector Database (Semantic search)

## AI Components

- Large Language Models (LLMs)
- Retrieval Augmented Generation (RAG)
- Vector Embeddings
- Multimodal AI (Image + Text)

---

# AI Features

## Image-Based Tourism Content Generation

Users can capture an image of a location and automatically generate tourism-related information.

Pipeline:

1. Image captured through mobile application
2. Image embeddings generated
3. Vector similarity search performed using Pinecone
4. Relevant location data retrieved
5. LLM generates structured tourism content

---

## Multilingual AI Chatbot

Provides tourism assistance in multiple languages and answers questions about locations, services, and travel guidance.

---

## Intelligent Trip Planning

Generates travel itineraries based on user preferences, location, and available attractions.

---

## Location-Aware Recommendations

Using **GPS and OSRM routing engine**, the system suggests Nearby attractions, Restaurants and services, Emergency contacts

---

# Tech Stack

### Programming Languages
- Python | JavaScript | TypeScript

### AI / Machine Learning
- Large Language Models (LLMs) | Retrieval Augmented Generation (RAG) | Vector Embeddings | Pinecone Vector Database | LangChain

### Backend
- FastAPI | Firebase

### Frontend
- Flutter | React.js

### Infrastructure
- Docker | AWS EC2

### APIs
- Google Gemini API | OSRM Routing Engine | Google map API

---

# Demo

Demo Video:  
(Add YouTube or Google Drive link)

---

