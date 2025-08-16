
# Python implementation of an Agentic AI for Marketing Teams

import time
import hashlib
import random
import logging
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from abc import ABC
from contextlib import contextmanager

import chromadb
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# --- Configuration and Data Models ---

# Load environment variables from .env file
load_dotenv()

@dataclass
class PersonalizationTraits:
    """Data model for customer personalization attributes."""
    preferred_brewing_method: str = "Pour-over"
    preferred_communication_tone: str = "sophisticated"

@dataclass
class CustomerSegment:
    """Data model for a customer segment from Salesforce."""
    name: str
    description: str
    targeting_criteria: str
    personalization_traits: PersonalizationTraits
    size: Optional[int] = None

@dataclass
class CreativeConcept:
    """Data model for a generated creative idea."""
    script_idea: str
    visual_keywords: List[str]
    audio_suggestion: str
    framework: str # e.g., 'storytelling', 'product_hero'

@dataclass
class CampaignBrief:
    """Data model for a marketing campaign brief."""
    campaign_goal: str
    target_audience_id: str
    key_message: str
    desired_tone_style: str
    call_to_action: str
    budget_tier: str = "standard"

@dataclass
class VideoOutput:
    """Data model for the final generated video output."""
    url: str
    metadata: Dict[str, Any]
    quality_score: float
    estimated_cost: float

# --- Enhanced Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('marketing_ai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Abstract Base Classes ---

class ExternalService(ABC):
    """Base class for all external service integrations."""
    def __init__(self):
        self.service_name = self.__class__.__name__
    
    @contextmanager
    def error_handling(self, operation_name: str):
        """Context manager for consistent error handling."""
        logger.info(f"[{self.service_name}] Starting: {operation_name}")
        try:
            yield
            logger.info(f"[{self.service_name}] Completed: {operation_name}")
        except requests.exceptions.HTTPError as e:
            logger.error(f"[{self.service_name}] HTTP Error during {operation_name}: {e.response.status_code} - {e.response.text}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"[{self.service_name}] Network error during {operation_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"[{self.service_name}] Unexpected error during {operation_name}: {e}")
            raise

class Agent(ABC):
    """Base class for all AI agents."""
    def __init__(self, name: str):
        self.name = name
    
    def log_decision(self, approved: bool, reason: str, score: float = 0.0):
        """Log agent decisions for performance monitoring."""
        decision = 'APPROVED' if approved else 'REJECTED'
        logger.info(f"[{self.name}] Decision: {decision} | Score: {score:.2f} | Reason: {reason}")

# --- External Service Integrations ---

class SalesforceMarketingCloudAPI(ExternalService):
    """Handles all communication with the Salesforce Marketing Cloud API."""
    def __init__(self):
        super().__init__()
        self.client_id = os.getenv("SFMC_CLIENT_ID")
        self.client_secret = os.getenv("SFMC_CLIENT_SECRET")
        self.auth_url = os.getenv("SFMC_AUTH_URL")
        self.base_api_url = os.getenv("SFMC_BASE_API_URL")
        self.access_token = None
        self.token_expires_at = 0

    def _get_access_token(self):
        """Authenticates with Salesforce and retrieves an access token."""
        if time.time() < self.token_expires_at:
            return # Token is still valid

        if not all([self.client_id, self.client_secret, self.auth_url]):
            logger.warning(f"[{self.service_name}] Missing Salesforce credentials. Running in mock mode.")
            self.access_token = "mock_access_token"
            self.token_expires_at = time.time() + 3600
            return

        with self.error_handling("Salesforce authentication"):
            payload = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }
            response = requests.post(self.auth_url, json=payload, timeout=30)
            response.raise_for_status()
            token_data = response.json()
            self.access_token = token_data["access_token"]
            self.token_expires_at = time.time() + token_data.get("expires_in", 3500)

    def get_customer_segment(self, segment_id: str) -> CustomerSegment:
        """Fetches customer segment data, including personalization traits."""
        self._get_access_token()
        
        # This section remains mocked as it requires a live Salesforce instance.
        # In a real application, you would replace this with actual API calls
        # to query Data Extensions or the Contact Model.
        logger.info(f"[{self.service_name}] Fetching C360 data for segment: {segment_id} (mocked)")
        
        segments = {
            "high_value_customers": CustomerSegment(
                name="High-Value Customers",
                description="Loyal customers with high lifetime value.",
                targeting_criteria="LTV > $500",
                personalization_traits=PersonalizationTraits(
                    preferred_brewing_method="Espresso",
                    preferred_communication_tone="exclusive"
                ),
                size=1250
            ),
            "new_subscribers": CustomerSegment(
                name="New Subscribers",
                description="Recent email subscribers exploring our brand.",
                targeting_criteria="Subscribed in last 30 days",
                personalization_traits=PersonalizationTraits(
                    preferred_brewing_method="Pour-over",
                    preferred_communication_tone="welcoming"
                ),
                size=890
            )
        }
        time.sleep(0.5) # Simulate API latency
        segment = segments.get(segment_id)
        if not segment:
            raise ValueError(f"Segment '{segment_id}' not found in mock data.")
        return segment

class VectorDatabase(ExternalService):
    """ChromaDB implementation for storing and searching brand guidelines."""
    def __init__(self):
        super().__init__()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client() # In-memory client
        self.collection = self.client.get_or_create_collection("brand_guidelines")
        self._initialize_brand_data()
        
    def _initialize_brand_data(self):
        """Loads brand guidelines into the vector database if it's empty."""
        if self.collection.count() > 0:
            return
        
        brand_guidelines = {
            "visual_luxury": "Emphasize premium materials, elegant surfaces, and sophisticated lighting.",
            "visual_authenticity": "Show real craftsmanship and genuine human interactions.",
            "messaging_storytelling": "Every piece of content should tell a story about origin, process, or connection.",
            "messaging_quality": "Highlight craftsmanship and premium ingredients without being boastful.",
            "personality_passionate": "Show genuine enthusiasm for coffee culture and flavor experiences.",
            "personality_welcoming": "Create inclusive content that invites customers into the community.",
            "audio_natural": "Prefer natural ambient sounds over produced music tracks.",
        }
        with self.error_handling("database initialization"):
            self.collection.add(
                documents=list(brand_guidelines.values()),
                ids=list(brand_guidelines.keys())
            )
            logger.info(f"[{self.service_name}] Loaded {self.collection.count()} brand guidelines.")
    
    def semantic_search(self, query_concepts: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """Performs semantic search and returns results with relevance scores."""
        query_text = ", ".join(query_concepts)
        with self.error_handling(f"semantic search for '{query_text}'"):
            results = self.collection.query(
                query_texts=[query_text], 
                n_results=min(top_k, self.collection.count())
            )
            matches = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    distance = results['distances'][0][i]
                    matches.append({
                        'id': results['ids'][0][i],
                        'content': doc,
                        'relevance_score': max(0, 1 - distance)
                    })
            matches.sort(key=lambda x: x['relevance_score'], reverse=True)
            return matches

class VideoGenerationAPI(ExternalService):
    """Simulates the Veo 3 API for video generation."""
    def generate_video(self, prompt: str, budget_tier: str) -> VideoOutput:
        """Generates a video based on a prompt, returning structured output."""
        with self.error_handling("video generation"):
            logger.info(f"[{self.service_name}] Initiating video generation (tier: {budget_tier})")
            
            # Simulate generation time and cost
            complexity = min(len(prompt.split()) / 150.0, 1.0) # Simple complexity metric
            generation_time = 2 + (complexity * 5)
            time.sleep(generation_time)
            
            video_id = hashlib.sha256(prompt.encode()).hexdigest()[:12]
            quality_score = {"standard": 0.75, "premium": 0.9}.get(budget_tier, 0.7) + (complexity * 0.1)
            estimated_cost = {"standard": 5.00, "premium": 12.00}.get(budget_tier, 5.0) * (1 + complexity)
            
            return VideoOutput(
                url=f"https://fake-veo-api.com/videos/{video_id}",
                metadata={"video_id": video_id, "duration": 8, "resolution": "720p"},
                quality_score=min(quality_score, 1.0),
                estimated_cost=estimated_cost
            )

# --- AI Agents ---

class BrandGuardianAgent(Agent):
    """Evaluates creative concepts against brand guidelines using semantic search."""
    def __init__(self, vector_db: VectorDatabase):
        super().__init__("Brand Guardian")
        self.vector_db = vector_db
        self.approval_threshold = 0.65

    def evaluate_concept(self, concept: CreativeConcept) -> tuple[bool, float, str]:
        """Evaluates a concept, returning approval status, score, and feedback."""
        guidelines = self.vector_db.semantic_search(concept.visual_keywords)
        if not guidelines:
            reason = "Concept does not align with any core brand guidelines."
            self.log_decision(False, reason)
            return False, 0.0, reason
        
        alignment_score = sum(g['relevance_score'] for g in guidelines) / len(guidelines)
        approved = alignment_score >= self.approval_threshold
        feedback = f"Strongest alignment with: '{guidelines[0]['id']}'." if guidelines else "No strong alignment."
        
        self.log_decision(approved, feedback, alignment_score)
        return approved, alignment_score, feedback

class CreativeDirectorAgent(Agent):
    """Crafts detailed, optimized video prompts from approved concepts."""
    def __init__(self):
        super().__init__("Creative Director")

    def craft_video_prompt(self, brief: CampaignBrief, concept: CreativeConcept, segment: CustomerSegment) -> str:
        """Creates a detailed video generation prompt."""
        logger.info(f"[{self.name}] Crafting prompt for '{concept.framework}' concept.")
        
        # Enhance scene description with personalization
        brewing_method = segment.personalization_traits.preferred_brewing_method
        enhanced_script = concept.script_idea.replace("{brewing_method}", brewing_method)

        prompt = f"""
Create an 8-second, 720p video with a cinematic, {brief.desired_tone_style} feel.
**Visual Style:** Emphasize {', '.join(concept.visual_keywords)}. Use rich, warm tones and natural, soft lighting.
**Scene Description:** {enhanced_script}
**Audio:** {concept.audio_suggestion}
"""
        return prompt.strip()

class ConceptGeneratorAgent(Agent):
    """Generates diverse creative concepts based on different frameworks."""
    def __init__(self):
        super().__init__("Concept Generator")

    def generate_concept(self, brief: CampaignBrief, segment: CustomerSegment) -> CreativeConcept:
        """Generates a single, personalized creative concept."""
        # In a real system, this could use an LLM to generate more varied ideas.
        # Here, we'll use a rule-based approach for demonstration.
        tone = segment.personalization_traits.preferred_communication_tone
        
        if tone == "exclusive":
            return self._product_hero_approach(brief)
        else:
            return self._storytelling_approach(brief)

    def _storytelling_approach(self, brief: CampaignBrief) -> CreativeConcept:
        """Creates a narrative-driven concept."""
        return CreativeConcept(
            script_idea="A quiet morning ritual unfolds, witnessing the meditative process of {brewing_method} brewing, celebrating a mindful moment.",
            visual_keywords=["ritual", "mindfulness", "process", "tranquility"],
            audio_suggestion="Gentle acoustic soundtrack with natural coffee preparation sounds.",
            framework="storytelling"
        )

    def _product_hero_approach(self, brief: CampaignBrief) -> CreativeConcept:
        """Creates a product-focused concept."""
        return CreativeConcept(
            script_idea="Extreme close-ups reveal the unique characteristics of our beans, followed by the precise {brewing_method} brewing process that honors their origin.",
            visual_keywords=["macro", "texture", "precision", "transformation"],
            audio_suggestion="Minimal music with enhanced coffee preparation sounds.",
            framework="product_hero"
        )

# --- Workflow Orchestrator ---

class MarketingWorkflowOrchestrator:
    """Manages the end-to-end workflow from brief to video generation."""
    def __init__(self):
        self.salesforce = SalesforceMarketingCloudAPI()
        self.vector_db = VectorDatabase()
        self.video_api = VideoGenerationAPI()
        self.concept_generator = ConceptGeneratorAgent()
        self.brand_guardian = BrandGuardianAgent(self.vector_db)
        self.creative_director = CreativeDirectorAgent()

    def execute_campaign(self, brief: CampaignBrief) -> Optional[Dict[str, Any]]:
        """Executes the full campaign workflow."""
        logger.info(f"--- Starting Campaign: {brief.campaign_goal} ---")
        try:
            segment = self.salesforce.get_customer_segment(brief.target_audience_id)
            concept = self.concept_generator.generate_concept(brief, segment)
            
            approved, score, feedback = self.brand_guardian.evaluate_concept(concept)
            if not approved:
                logger.error("Campaign halted: Concept rejected by Brand Guardian.")
                return None
            
            video_prompt = self.creative_director.craft_video_prompt(brief, concept, segment)
            video_output = self.video_api.generate_video(video_prompt, brief.budget_tier)
            
            results = {
                "status": "success",
                "brief": asdict(brief),
                "segment": asdict(segment),
                "video_output": asdict(video_output)
            }
            logger.info(f"--- Campaign Successful ---")
            logger.info(f"Video URL: {video_output.url}")
            logger.info(f"Estimated Cost: ${video_output.estimated_cost:.2f}")
            return results

        except Exception as e:
            logger.error(f"Campaign execution failed catastrophically: {e}")
            return None

# --- Main Execution ---

def main():
    """Main function to run campaign scenarios."""
    orchestrator = MarketingWorkflowOrchestrator()

    # Scenario 1: Premium launch for high-value customers
    brief1 = CampaignBrief(
        campaign_goal="Launch new Ethiopian single-origin beans",
        target_audience_id="high_value_customers",
        key_message="Experience the unique terroir of our latest discovery.",
        desired_tone_style="sophisticated and educational",
        call_to_action="Available now in limited quantities.",
        budget_tier="premium"
    )
    orchestrator.execute_campaign(brief1)

    # Scenario 2: Welcoming new subscribers
    brief2 = CampaignBrief(
        campaign_goal="Welcome new subscribers to the brand",
        target_audience_id="new_subscribers",
        key_message="Begin your journey with exceptional coffee.",
        desired_tone_style="welcoming and informative",
        call_to_action="Explore our curated selection."
    )
    orchestrator.execute_campaign(brief2)


if __name__ == "__main__":
    main()