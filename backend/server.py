from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime

# AI agents
from ai_agents.agents import AgentConfig, SearchAgent, ChatAgent


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# AI agents init
agent_config = AgentConfig()
search_agent: Optional[SearchAgent] = None
chat_agent: Optional[ChatAgent] = None

# Main app
app = FastAPI(title="AI Agents API", description="Minimal AI Agents API with LangGraph and MCP support")

# API router
api_router = APIRouter(prefix="/api")


# Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

# LLM Models for Hot-or-Not
class LLMModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    provider: str
    description: str
    parameters: str
    avatar_url: Optional[str] = None
    rating: float = Field(default=1400.0)  # ELO rating starting at 1400
    votes: int = Field(default=0)
    wins: int = Field(default=0)
    losses: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class LLMModelCreate(BaseModel):
    name: str
    provider: str
    description: str
    parameters: str
    avatar_url: Optional[str] = None

class VoteRequest(BaseModel):
    winner_id: str
    loser_id: str

class VoteResponse(BaseModel):
    success: bool
    winner_new_rating: float
    loser_new_rating: float
    message: str


# AI agent models
class ChatRequest(BaseModel):
    message: str
    agent_type: str = "chat"  # "chat" or "search"
    context: Optional[dict] = None


class ChatResponse(BaseModel):
    success: bool
    response: str
    agent_type: str
    capabilities: List[str]
    metadata: dict = Field(default_factory=dict)
    error: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    max_results: int = 5


class SearchResponse(BaseModel):
    success: bool
    query: str
    summary: str
    search_results: Optional[dict] = None
    sources_count: int
    error: Optional[str] = None

# Routes
@api_router.get("/")
async def root():
    return {"message": "Hello World"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]


# LLM Hot-or-Not Routes
@api_router.post("/models", response_model=LLMModel)
async def create_model(model: LLMModelCreate):
    model_dict = model.dict()
    model_obj = LLMModel(**model_dict)
    await db.llm_models.insert_one(model_obj.dict())
    return model_obj

@api_router.get("/models", response_model=List[LLMModel])
async def get_all_models():
    models = await db.llm_models.find().to_list(1000)
    return [LLMModel(**model) for model in models]

@api_router.get("/models/random")
async def get_random_pair():
    # Get two random models for comparison
    models = await db.llm_models.aggregate([{"$sample": {"size": 2}}]).to_list(2)
    if len(models) < 2:
        raise HTTPException(status_code=400, detail="Not enough models for comparison")
    return [LLMModel(**model) for model in models]

@api_router.get("/models/leaderboard")
async def get_leaderboard():
    # Get top models by rating
    models = await db.llm_models.find().sort("rating", -1).limit(20).to_list(20)
    return [LLMModel(**model) for model in models]

def calculate_elo_rating(winner_rating: float, loser_rating: float, k_factor: int = 32) -> tuple:
    """Calculate new ELO ratings after a match"""
    expected_winner = 1 / (1 + 10**((loser_rating - winner_rating) / 400))
    expected_loser = 1 / (1 + 10**((winner_rating - loser_rating) / 400))

    new_winner_rating = winner_rating + k_factor * (1 - expected_winner)
    new_loser_rating = loser_rating + k_factor * (0 - expected_loser)

    return new_winner_rating, new_loser_rating

@api_router.post("/vote", response_model=VoteResponse)
async def vote_on_models(vote: VoteRequest):
    # Get both models
    winner = await db.llm_models.find_one({"id": vote.winner_id})
    loser = await db.llm_models.find_one({"id": vote.loser_id})

    if not winner or not loser:
        raise HTTPException(status_code=404, detail="One or both models not found")

    # Calculate new ratings
    winner_rating = winner.get("rating", 1400.0)
    loser_rating = loser.get("rating", 1400.0)

    new_winner_rating, new_loser_rating = calculate_elo_rating(winner_rating, loser_rating)

    # Update winner
    await db.llm_models.update_one(
        {"id": vote.winner_id},
        {
            "$set": {"rating": new_winner_rating},
            "$inc": {"votes": 1, "wins": 1}
        }
    )

    # Update loser
    await db.llm_models.update_one(
        {"id": vote.loser_id},
        {
            "$set": {"rating": new_loser_rating},
            "$inc": {"votes": 1, "losses": 1}
        }
    )

    return VoteResponse(
        success=True,
        winner_new_rating=new_winner_rating,
        loser_new_rating=new_loser_rating,
        message=f"{winner['name']} wins!"
    )

@api_router.post("/models/seed")
async def seed_models():
    # Check if models already exist
    existing_count = await db.llm_models.count_documents({})
    if existing_count > 0:
        return {"message": f"Models already seeded. Found {existing_count} models."}

    # Popular LLM models to seed
    models_to_seed = [
        {
            "name": "GPT-4",
            "provider": "OpenAI",
            "description": "OpenAI's most capable model with advanced reasoning and instruction following",
            "parameters": "~1.7T parameters",
            "avatar_url": "https://images.unsplash.com/photo-1677442136019-21780ecad995?w=400&h=400&fit=crop&crop=face"
        },
        {
            "name": "Claude 3.5 Sonnet",
            "provider": "Anthropic",
            "description": "Anthropic's flagship model with excellent reasoning and creative capabilities",
            "parameters": "~175B parameters",
            "avatar_url": "https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=400&h=400&fit=crop&crop=face"
        },
        {
            "name": "Gemini Pro",
            "provider": "Google",
            "description": "Google's advanced multimodal model with strong analytical capabilities",
            "parameters": "~540B parameters",
            "avatar_url": "https://images.unsplash.com/photo-1535303311164-664fc9ec6532?w=400&h=400&fit=crop&crop=face"
        },
        {
            "name": "Llama 2 70B",
            "provider": "Meta",
            "description": "Meta's open-source model with strong performance across diverse tasks",
            "parameters": "70B parameters",
            "avatar_url": "https://images.unsplash.com/photo-1599566150163-29194dcaad36?w=400&h=400&fit=crop&crop=face"
        },
        {
            "name": "PaLM 2",
            "provider": "Google",
            "description": "Google's language model with improved reasoning and coding abilities",
            "parameters": "~340B parameters",
            "avatar_url": "https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?w=400&h=400&fit=crop&crop=face"
        },
        {
            "name": "Mistral 7B",
            "provider": "Mistral AI",
            "description": "High-performance open model with efficient architecture",
            "parameters": "7B parameters",
            "avatar_url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop&crop=face"
        },
        {
            "name": "ChatGPT",
            "provider": "OpenAI",
            "description": "Popular conversational AI with broad knowledge and capabilities",
            "parameters": "~175B parameters",
            "avatar_url": "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=400&h=400&fit=crop&crop=face"
        },
        {
            "name": "Cohere Command",
            "provider": "Cohere",
            "description": "Enterprise-focused model with strong business and analytical capabilities",
            "parameters": "~52B parameters",
            "avatar_url": "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=400&h=400&fit=crop&crop=face"
        },
        {
            "name": "Claude 2",
            "provider": "Anthropic",
            "description": "Anthropic's previous generation model with strong safety focus",
            "parameters": "~130B parameters",
            "avatar_url": "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=400&h=400&fit=crop&crop=face"
        },
        {
            "name": "GPT-3.5 Turbo",
            "provider": "OpenAI",
            "description": "Fast and efficient model optimized for chat applications",
            "parameters": "~175B parameters",
            "avatar_url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop&crop=face"
        }
    ]

    # Create model objects and insert
    created_models = []
    for model_data in models_to_seed:
        model_obj = LLMModel(**model_data)
        await db.llm_models.insert_one(model_obj.dict())
        created_models.append(model_obj)

    return {"message": f"Successfully seeded {len(created_models)} models", "models": created_models}


# AI agent routes
@api_router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    # Chat with AI agent
    global search_agent, chat_agent
    
    try:
        # Init agents if needed
        if request.agent_type == "search" and search_agent is None:
            search_agent = SearchAgent(agent_config)
            
        elif request.agent_type == "chat" and chat_agent is None:
            chat_agent = ChatAgent(agent_config)
        
        # Select agent
        agent = search_agent if request.agent_type == "search" else chat_agent
        
        if agent is None:
            raise HTTPException(status_code=500, detail="Failed to initialize agent")
        
        # Execute agent
        response = await agent.execute(request.message)
        
        return ChatResponse(
            success=response.success,
            response=response.content,
            agent_type=request.agent_type,
            capabilities=agent.get_capabilities(),
            metadata=response.metadata,
            error=response.error
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return ChatResponse(
            success=False,
            response="",
            agent_type=request.agent_type,
            capabilities=[],
            error=str(e)
        )


@api_router.post("/search", response_model=SearchResponse)
async def search_and_summarize(request: SearchRequest):
    # Web search with AI summary
    global search_agent
    
    try:
        # Init search agent if needed
        if search_agent is None:
            search_agent = SearchAgent(agent_config)
        
        # Search with agent
        search_prompt = f"Search for information about: {request.query}. Provide a comprehensive summary with key findings."
        result = await search_agent.execute(search_prompt, use_tools=True)
        
        if result.success:
            return SearchResponse(
                success=True,
                query=request.query,
                summary=result.content,
                search_results=result.metadata,
                sources_count=result.metadata.get("tools_used", 0)
            )
        else:
            return SearchResponse(
                success=False,
                query=request.query,
                summary="",
                sources_count=0,
                error=result.error
            )
            
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        return SearchResponse(
            success=False,
            query=request.query,
            summary="",
            sources_count=0,
            error=str(e)
        )


@api_router.get("/agents/capabilities")
async def get_agent_capabilities():
    # Get agent capabilities
    try:
        capabilities = {
            "search_agent": SearchAgent(agent_config).get_capabilities(),
            "chat_agent": ChatAgent(agent_config).get_capabilities()
        }
        return {
            "success": True,
            "capabilities": capabilities
        }
    except Exception as e:
        logger.error(f"Error getting capabilities: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# Include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    # Initialize agents on startup
    global search_agent, chat_agent
    logger.info("Starting AI Agents API...")
    
    # Lazy agent init for faster startup
    logger.info("AI Agents API ready!")


@app.on_event("shutdown")
async def shutdown_db_client():
    # Cleanup on shutdown
    global search_agent, chat_agent
    
    # Close MCP
    if search_agent and search_agent.mcp_client:
        # MCP cleanup automatic
        pass
    
    client.close()
    logger.info("AI Agents API shutdown complete.")
