# Agentic AI for Personalized Video Marketing

This project is a Python-based framework for an agentic AI system designed to automate the creation of personalized marketing videos via MCP. It connects to Salesforce Marketing Cloud for customer data, uses a vector database for brand safety, and orchestrates a team of AI agents to generate creative concepts and video prompts for a luxury coffee brand. It can be adapted for any kind of brand provided a well thought out definition. It's designed to be modular and as definitions grow, videos become more true to brand. 

Different agentic roles work together to create organization-sanctioned content mimicking a team of staff working together. When complete, output to the console is logged, making decisions clear to end users and enabling additional prompting and future edge case handling as necessary.

As Diffusion models get better with each new release, generative video becomes a more feasible sales and marketing tool. It's highly likely that we will see completely custom automated mixed-media campaigns in the very near future.

## Features
Multi-Agent System: Utilizes distinct AI agents (Concept Generator, Brand Guardian, Creative Director) each with a specific role in the creative workflow.

Deep Personalization: Connects to Salesforce Marketing Cloud to fetch Customer 360 data, enabling subtle, on-brand personalization of video content based on audience preferences.

Brand Safety & Consistency: Employs a vector database (ChromaDB) with semantic search to ensure all creative concepts are rigorously checked against brand guidelines.

Automated Creative Ideation: Generates diverse creative concepts using different strategic frameworks (e.g., storytelling, product-focused).

Modular & Extensible: Built with abstract base classes, making it easy to add new agents, services, or video generation APIs.

Secure Configuration: Manages all API keys and credentials securely using a .env file - populated with your credentials.

## Architecture
The system is built around a central Orchestrator that manages the flow of information between specialized AI agents and external services.

Campaign Brief: A marketing manager provides a high-level campaign brief.

Salesforce API: The orchestrator fetches detailed customer segment data, including personalization traits.

Concept Generator Agent: Creates a personalized creative concept based on the brief and customer data.

Brand Guardian Agent: Uses a vector database to perform a semantic search, ensuring the concept aligns with brand guidelines. If rejected, the process can be halted or retried.

Creative Director Agent: If the concept is approved, this agent crafts a detailed, optimized prompt for the video generation API.

Video Generation API: A simulated API (e.g., Google Veo 3) generates the final video based on the prompt.



The script will run two pre-defined campaign scenarios, logging the entire decision-making process of the AI agents to both the console and a marketing_ai.log file.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
