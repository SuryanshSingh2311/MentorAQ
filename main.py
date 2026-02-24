# main.py - Production-Ready FastAPI Backend with LangChain & MongoDB

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, ValidationError
from typing import List, Dict, Any, Optional
from datetime import datetime
import tempfile
import os
import json
import re
from io import BytesIO
from contextlib import asynccontextmanager
import logging

# MongoDB imports
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from bson.errors import InvalidId

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    print("WARNING: python-dotenv not installed. Using system environment variables only.")
except Exception as e:
    print(f"WARNING: Error loading .env file: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Global Resource Storage ---
llm_model: Optional[ChatHuggingFace] = None
embeddings_model: Optional[HuggingFaceEmbeddings] = None
mentor_retriever: Optional[Any] = None

# MongoDB Globals
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "mentoraq_db")
mongo_client: Optional[AsyncIOMotorClient] = None
db = None

# --- Database Setup ---

async def init_db():
    """Initialize MongoDB connection and seed sample data."""
    global mongo_client, db
    try:
        logger.info(f"Connecting to MongoDB at {MONGO_URL}...")
        mongo_client = AsyncIOMotorClient(MONGO_URL)
        db = mongo_client[DB_NAME]
        
        # Verify connection
        await mongo_client.admin.command('ping')
        logger.info("MongoDB connected successfully")
        
        # Check if we need to seed data
        faculty_count = await db.faculty.count_documents({})
        if faculty_count == 0:
            sample_faculty = [
                {
                    "name": "Dr. Aditi Sharma", "title": "Associate Professor", "department": "CSE", 
                    "email": "aditi.sharma@example.edu", "expertise": "Machine Learning, NLP, Python, Deep Learning", 
                    "subjects": "ML, Data Structures, AI", "research_areas": "NLP, Knowledge Graphs, Computer Vision", 
                    "bio": "Works on practical ML for education and industry applications.", "availability": "Mon/Wed 3-5pm"
                },
                {
                    "name": "Prof. Rajiv Verma", "title": "Professor", "department": "Mechanical", 
                    "email": "r.verma@example.edu", "expertise": "Thermodynamics, Robotics, MATLAB, Control Systems", 
                    "subjects": "Thermo, Robotics, Mechanics", "research_areas": "Robotics control systems, Automation", 
                    "bio": "Industry-oriented research in robotics and automation.", "availability": "Tue/Thu 10-12pm"
                },
                {
                    "name": "Dr. Meera Iyer", "title": "Assistant Professor", "department": "Electronics", 
                    "email": "meera.iyer@example.edu", "expertise": "Embedded Systems, VLSI, C/C++, IoT", 
                    "subjects": "Circuits, Embedded, Digital Design", "research_areas": "Low-power VLSI, IoT Security", 
                    "bio": "Embedded systems and IoT devices for smart applications.", "availability": "Fri 2-4pm"
                },
                {
                    "name": "Dr. Ankit Gupta", "title": "Assistant Professor", "department": "CSE", 
                    "email": "ankit.gupta@example.edu", "expertise": "Web Development, Full Stack, JavaScript, React", 
                    "subjects": "Web Tech, Databases, Software Engineering", "research_areas": "Cloud Computing, DevOps", 
                    "bio": "Focuses on modern web technologies and cloud solutions.", "availability": "Mon/Fri 1-3pm"
                }
            ]
            await db.faculty.insert_many(sample_faculty)
            logger.info(f"Seeded {len(sample_faculty)} faculty records to MongoDB")
            
    except Exception as e:
        logger.error(f"MongoDB initialization failed: {e}")
        raise

async def get_database():
    """Dependency that yields the MongoDB database instance."""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    return db

# --- Pydantic Models ---

class RawLLMAnalysis(BaseModel):
    ats_score: int = Field(description="Compatibility score between 0 and 100", ge=0, le=100)
    strengths: List[str] = Field(description="5 or more key strengths from the resume")
    weaknesses: List[str] = Field(description="5 or more areas for improvement")
    keywords: List[str] = Field(description="10-15 important skills and technologies")
    analysis_summary: str = Field(description="Concise overall analysis summary")
    improvements: List[str] = Field(description="5 to 7 specific, actionable improvements to make") 


class ResumeAnalysisResponse(BaseModel):
    ats_score: int
    analysis: str
    keywords: List[str]
    improvements: str
    strengths: List[str]
    weaknesses: List[str]

class SkillsRequest(BaseModel):
    skills: List[str]
    career_interest: Optional[str] = ""

class InternshipRecommendation(BaseModel):
    id: int
    title: str
    company: str
    location: str
    duration: str
    stipend: str
    required_skills: List[str]
    description: str
    match_score: float
    apply_link: str

class CourseRecommendation(BaseModel):
    id: int
    title: str
    platform: str
    instructor: str
    duration: str
    level: str
    skills_covered: List[str]
    rating: float
    enrolled: str
    price: str
    description: str
    match_score: float
    link: str

class RecommendationsResponse(BaseModel):
    internships: List[InternshipRecommendation]
    courses: List[CourseRecommendation]

class MentorSearchRequest(BaseModel):
    query: str
    search_type: str = "keyword"

class Faculty(BaseModel):
    id: str # Changed to string for MongoDB ObjectId
    name: str
    title: str
    department: str
    email: str
    expertise: str
    subjects: str
    research_areas: str
    bio: str
    availability: str

class MentorshipRequest(BaseModel):
    student_name: str
    student_email: EmailStr
    faculty_id: str # Changed to string for MongoDB ObjectId
    message: str

class RecommendationsLLMResponse(BaseModel):
    internships: List[InternshipRecommendation]
    courses: List[CourseRecommendation]

class RoadmapStep(BaseModel):
    week: int
    title: str
    description: str
    resources: List[str]

class RoadmapResponse(BaseModel):
    goal: str
    duration: int
    steps: List[RoadmapStep]

class RoadmapRequest(BaseModel):
    goal: str
    skills: str
    duration: int

class InterviewStartRequest(BaseModel):
    user_email: str
    field: str
    topic: str
    difficulty: str

class InterviewStartResponse(BaseModel):
    session_id: str
    question: str

class InterviewAnswerRequest(BaseModel):
    session_id: str
    answer: str

class InterviewAnswerResponse(BaseModel):
    feedback: str
    next_question: str

class InterviewEndRequest(BaseModel):
    session_id: str

class ScorecardResponse(BaseModel):
    overall_score: int
    strengths: List[str]
    weaknesses: List[str]
    detailed_feedback: str

class OutreachRequest(BaseModel):
    skills_and_strengths: str
    target_role: str
    target_company: str

class OutreachResponse(BaseModel):
    email_formal: str
    linkedin_dm: str
    email_short: str

def _initialize_llm():
    global llm_model
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if api_token:
        logger.info(f"DEBUG: Token being used starts with: {api_token[:8]}...")
    if not api_token:
        logger.error("HUGGINGFACEHUB_API_TOKEN not found in environment variables")
        return
        
    try:
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.2-3B-Instruct",
            task="text-generation",
            max_new_tokens=1024,
            temperature=0.7,
            huggingfacehub_api_token=api_token
        )
        llm_model = ChatHuggingFace(llm=llm)
        logger.info("LLM model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LLM model: {e}")

def _initialize_embeddings():
   global embeddings_model
   try:
        # 1. Grab the valid token we know is working
        api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        
        # 2. Force the environment variable that sentence-transformers looks for
        if api_token:
            os.environ["HF_TOKEN"] = api_token
            
        # 3. Initialize the model, explicitly passing the token just to be bulletproof
        embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            cache_folder=os.getenv("EMBEDDINGS_CACHE", "./embeddings_cache"),
            model_kwargs={"token": api_token} 
        )
        logger.info("Embeddings model initialized successfully")
   except Exception as e:
        logger.error(f"Failed to initialize embeddings model: {e}")

async def _initialize_mentor_retriever():
    """Initialize Chroma vector store fetching from MongoDB."""
    global mentor_retriever, embeddings_model, db
    
    if embeddings_model is None:
        logger.warning("Embeddings model not loaded, skipping Chroma initialization.")
        return
    
    try:
        logger.info("Initializing Chroma vector store from MongoDB...")
        
        cursor = db.faculty.find({})
        all_faculty = await cursor.to_list(length=None)
        
        docs = []
        for faculty in all_faculty:
            content = (
                f"Name: {faculty['name']}, Department: {faculty['department']}\n"
                f"Expertise: {faculty['expertise']}\n"
                f"Research Areas: {faculty['research_areas']}\n"
                f"Bio: {faculty['bio']}"
            )
            docs.append(Document(
                page_content=content, 
                metadata={'faculty_id': str(faculty['_id']), 'name': faculty['name']}
            ))
        
        if docs:
            persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
            vector_db = Chroma.from_documents(
                docs, 
                embeddings_model,
                persist_directory=persist_directory
            )
            mentor_retriever = vector_db.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 5}
            )
            logger.info(f"Chroma initialized with {len(docs)} faculty documents")
        else:
            logger.warning("No faculty data found for Chroma initialization")
            
    except Exception as e:
        logger.error(f"Failed to initialize mentor retriever: {e}")
        mentor_retriever = None

# --- Helper Functions ---

def extract_text_from_pdf(file_content: bytes) -> str:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in docs])
        
        if not full_text.strip():
            raise ValueError("No text content extracted from PDF")
        return full_text
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {tmp_path}: {e}")

def extract_json_from_text(text: str) -> str:
    """Extract JSON object from text by finding the outermost brackets."""
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return text

def calculate_skill_match(user_skills: List[str], required_skills: List[str]) -> float:
    if not required_skills: return 0.0
    user_skills_lower = [s.lower().strip() for s in user_skills]
    required_skills_lower = [s.lower().strip() for s in required_skills]
    matches = sum(1 for skill in required_skills_lower 
                  if any(user_skill in skill or skill in user_skill 
                         for user_skill in user_skills_lower))
    return (matches / len(required_skills)) * 100

def parse_mongo_doc(doc: dict) -> dict:
    """Helper to convert MongoDB _id to string id for frontend."""
    if doc and "_id" in doc:
        doc["id"] = str(doc.pop("_id"))
        # Handle embedded ObjectIds if any (like faculty_id in requests)
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                doc[key] = str(value)
    return doc

# --- Static Data ---
# (Keeping internships and courses static as per original architecture)
INTERNSHIPS_DATA = [
    {"id": 1, "title": "Machine Learning Intern", "company": "TechCorp AI", "location": "Remote", "duration": "3 months", "stipend": "₹15,000/month", "required_skills": ["Python", "Machine Learning", "TensorFlow", "Data Analysis"], "description": "Work on cutting-edge ML models.", "apply_link": "#"},
    {"id": 2, "title": "Full Stack Developer Intern", "company": "WebSolutions Inc", "location": "Bangalore", "duration": "6 months", "stipend": "₹20,000/month", "required_skills": ["JavaScript", "React", "Node.js", "MongoDB"], "description": "Build scalable web applications.", "apply_link": "#"}
]

COURSES_DATA = [
    {"id": 1, "title": "Advanced Machine Learning Specialization", "platform": "Coursera", "instructor": "Andrew Ng", "duration": "4 months", "level": "Advanced", "skills_covered": ["Deep Learning", "Neural Networks", "TensorFlow"], "rating": 4.8, "enrolled": "150K+", "price": "₹3,999", "description": "Master advanced ML techniques.", "link": "#"},
    {"id": 2, "title": "Full Stack Web Development Bootcamp", "platform": "Udemy", "instructor": "Colt Steele", "duration": "3 months", "level": "Intermediate", "skills_covered": ["HTML", "CSS", "JavaScript", "React", "Node.js"], "rating": 4.7, "enrolled": "200K+", "price": "₹1,499", "description": "Become a full-stack developer.", "link": "#"}
]

# --- FastAPI Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application initialization...")
    try:
        await init_db()
        _initialize_llm()
        _initialize_embeddings()
        await _initialize_mentor_retriever()
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"FATAL: Application startup failed: {e}")
        raise
    
    yield
    
    logger.info("Closing MongoDB connection...")
    if mongo_client:
        mongo_client.close()
    logger.info("Application shutdown complete")

# --- FastAPI App ---

app = FastAPI(
    title="MentorAQ API",
    description="Production-ready AI-powered career mentorship platform",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---

@app.get("/")
async def root():
    return {
        "message": "MentorAQ API - Production Ready (MongoDB)",
        "status": "online",
        "database": "connected" if db is not None else "disconnected"
    }
@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy" if llm_model else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": db is not None,
            "llm_model": llm_model is not None,
            "embeddings_model": embeddings_model is not None,
            "mentor_retriever": mentor_retriever is not None
        }
    }

@app.post("/api/analyze-resume", response_model=ResumeAnalysisResponse)
async def analyze_resume(
    file: UploadFile = File(...),
    job_description: str = Form("")
):
    """Analyze resume for ATS compatibility using LLM."""
    if llm_model is None:
        raise HTTPException(status_code=503, detail="AI service unavailable. Please try again later.")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        file_content = await file.read()
        resume_text = extract_text_from_pdf(file_content)
        
        parser = JsonOutputParser(pydantic_object=RawLLMAnalysis)
        
        # Simplified prompt: "Output ONLY JSON"
        analysis_prompt = PromptTemplate(
            template="""You are an expert ATS (Applicant Tracking System) analyst. Analyze the following resume against the job description (if provided).

Resume Text:
{resume_text}

Job Description:
{job_description}

{format_instructions}
IMPORTANT: Output ONLY valid JSON. Do not include any conversational text, introductions, or markdown formatting (like ```json). Start your response immediately with {{ and end with }}.
""",
            input_variables=['resume_text', 'job_description'],
            partial_variables={'format_instructions': parser.get_format_instructions()}
        )
        
        chain = analysis_prompt | llm_model | StrOutputParser()
        raw_response = chain.invoke({
            'resume_text': resume_text[:4000],
            'job_description': job_description[:1500] if job_description else "No specific job description provided."
        })
        
        logger.info(f"Raw LLM Response: {raw_response[:200]}...") # Helpful for debugging
        
        try:
            # Extract and Parse
            json_str = extract_json_from_text(raw_response)
            raw_parsed_data = parser.parse(json_str) 
            
            if isinstance(raw_parsed_data, dict):
                parsed_analysis = RawLLMAnalysis(**raw_parsed_data) 
            elif isinstance(raw_parsed_data, RawLLMAnalysis):
                parsed_analysis = raw_parsed_data
            else:
                 raise TypeError("Unexpected parser return type")
            
            # Convert the list of improvements into a nicely formatted string for your frontend UI
            formatted_improvements = "<br>".join([f"• {imp}" for imp in parsed_analysis.improvements])
                 
        except (Exception, ValidationError) as parse_error:
             logger.error(f"Failed to parse LLM output: {parse_error}\nRaw output was: {raw_response}")
             return ResumeAnalysisResponse(
                 ats_score=65,
                 analysis="Unable to complete full analysis due to unexpected LLM response format.",
                 keywords=["Python", "JavaScript", "SQL", "Communication", "Problem Solving"],
                 improvements="Add more specific technical skills and quantify achievements.",
                 strengths=["Clear structure"],
                 weaknesses=["Technical skills section needs expansion"]
             )
        
        return ResumeAnalysisResponse(
            ats_score=parsed_analysis.ats_score,
            analysis=parsed_analysis.analysis_summary,
            keywords=parsed_analysis.keywords,
            improvements=formatted_improvements,
            strengths=parsed_analysis.strengths,
            weaknesses=parsed_analysis.weaknesses
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resume analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/api/recommendations", response_model=RecommendationsResponse)
async def get_recommendations(request: SkillsRequest):
    """Get AI-generated personalized internship and course recommendations."""
    if llm_model is None:
        raise HTTPException(status_code=503, detail="AI service unavailable. Please try again later.")
        
    try:
        user_skills_str = ", ".join(request.skills)
        career_interest = request.career_interest or "General Tech and Engineering"
        
        parser = JsonOutputParser(pydantic_object=RecommendationsLLMResponse)
        
        recommendation_prompt = PromptTemplate(
            template="""You are an expert career counselor. 
            Based on the user's current skills and career interest, generate 3 highly relevant, realistic (but fictional) internships and 3 relevant online courses.
            
            User Skills: {skills}
            Career Interest: {interest}
            
            {format_instructions}
            IMPORTANT: Output ONLY valid JSON. Do not include any conversational text. Start immediately with {{ and end with }}.
            """,
            input_variables=['skills', 'interest'],
            partial_variables={'format_instructions': parser.get_format_instructions()}
        )
        
        chain = recommendation_prompt | llm_model | StrOutputParser()
        raw_response = chain.invoke({
            'skills': user_skills_str,
            'interest': career_interest
        })
        
        logger.info(f"Raw Recommendations Response: {raw_response[:200]}...")
        
        json_str = extract_json_from_text(raw_response)
        parsed_data = parser.parse(json_str)
        
        import random
        
        clean_internships = []
        for i, item in enumerate(parsed_data.get('internships', [])):
            clean_internships.append(InternshipRecommendation(
                id=i + 1,
                title=item.get('title', 'Technology Intern'),
                # Fallback to 'platform' if LLM confused course/internship schemas
                company=item.get('company', item.get('platform', 'Tech Solutions Inc.')),
                location=item.get('location', 'Remote'),
                duration=item.get('duration', '3 months'),
                stipend=item.get('stipend', 'Variable'),
                required_skills=item.get('required_skills', item.get('skills_covered', request.skills)),
                description=item.get('description', 'A great opportunity to apply your skills in a real-world environment.'),
                match_score=random.randint(88, 98),
                apply_link=item.get('apply_link', 'https://linkedin.com/jobs')
            ))
            
        clean_courses = []
        for i, item in enumerate(parsed_data.get('courses', [])):
            clean_courses.append(CourseRecommendation(
                id=i + 1,
                title=item.get('title', 'Advanced Skill Course'),
                platform=item.get('platform', item.get('company', 'Online Learning Platform')),
                instructor=item.get('instructor', 'Expert Instructor'),
                duration=item.get('duration', '4 weeks'),
                level=item.get('level', 'Intermediate'),
                skills_covered=item.get('skills_covered', item.get('required_skills', request.skills)),
                rating=float(item.get('rating', 4.5)),
                enrolled=str(item.get('enrolled', '10k+')),
                price=str(item.get('price', 'Free to Audit')),
                description=item.get('description', 'Master these concepts with hands-on projects.'),
                match_score=random.randint(85, 95),
                link=item.get('link', 'https://coursera.org')
            ))
                
        return RecommendationsResponse(
            internships=clean_internships[:3],
            courses=clean_courses[:3]
        )
        
    except Exception as e:
        logger.error(f"AI Recommendations error: {e}. Falling back to static data.")
        # FALLBACK: Use static data if the LLM completely fails
        scored_internships = []
        for internship in INTERNSHIPS_DATA:
            match_score = calculate_skill_match(request.skills, internship['required_skills'])
            internship_copy = internship.copy()
            internship_copy['match_score'] = round(match_score, 2)
            scored_internships.append(InternshipRecommendation(**internship_copy))
        scored_internships.sort(key=lambda x: x.match_score, reverse=True)
        
        scored_courses = []
        for course in COURSES_DATA:
            match_score = calculate_skill_match(request.skills, course['skills_covered'])
            course_copy = course.copy()
            course_copy['match_score'] = round(match_score, 2)
            scored_courses.append(CourseRecommendation(**course_copy))
        scored_courses.sort(key=lambda x: x.match_score, reverse=True)
        
        return RecommendationsResponse(
            internships=scored_internships[:3],
            courses=scored_courses[:3]
        )

@app.post("/api/roadmap", response_model=RoadmapResponse)
async def generate_roadmap(request: RoadmapRequest):
    """Generate a step-by-step career/learning roadmap."""
    if llm_model is None:
        raise HTTPException(status_code=503, detail="AI service unavailable.")
        
    try:
        parser = JsonOutputParser(pydantic_object=RoadmapResponse)
        
        prompt = PromptTemplate(
            template="""You are a veteran technical mentor. Create a highly structured, step-by-step learning roadmap for a student.
            
            Goal: {goal}
            Current Skills: {skills}
            Duration: {duration} weeks
            
            Divide the learning plan into exactly {duration} weeks.
            {format_instructions}
            IMPORTANT: Output ONLY valid JSON. Do not include any conversational text or markdown. Start immediately with {{ and end with }}.
            """,
            input_variables=['goal', 'skills', 'duration'],
            partial_variables={'format_instructions': parser.get_format_instructions()}
        )
        
        chain = prompt | llm_model | StrOutputParser()
        raw_response = chain.invoke({
            'goal': request.goal,
            'skills': request.skills if request.skills else "Beginner",
            'duration': request.duration
        })
        
        json_str = extract_json_from_text(raw_response)
        parsed_data = parser.parse(json_str)
        
        # --- BULLETPROOF PARSING ---
        clean_steps = []
        for i, step in enumerate(parsed_data.get('steps', [])):
            # Force exactly the requested duration
            if i >= request.duration: 
                break
                
            clean_steps.append(RoadmapStep(
                week=i + 1,
                title=step.get('title', f'Week {i+1} Focus'),
                description=step.get('description', 'Focus on core fundamentals and hands-on practice.'),
                # Ensure resources is always a list
                resources=step.get('resources', ['Official Documentation', 'YouTube Tutorials']) if isinstance(step.get('resources'), list) else ['Official Documentation']
            ))
            
        return RoadmapResponse(
            goal=parsed_data.get('goal', request.goal),
            duration=len(clean_steps),
            steps=clean_steps
        )
        
    except Exception as e:
        logger.error(f"Roadmap generation error: {e}. Falling back to default.")
        # FALLBACK: Provide generic structured data if the LLM completely fails
        fallback_steps = [
            RoadmapStep(
                week=i+1, 
                title=f"Week {i+1}: Core Concepts", 
                description="Review basics, read documentation, and build a small practical project.", 
                resources=["FreeCodeCamp", "MDN Web Docs"]
            )
            for i in range(request.duration)
        ]
        return RoadmapResponse(goal=request.goal, duration=request.duration, steps=fallback_steps)

@app.post("/api/mentors/search")
async def search_mentors(request: MentorSearchRequest, database = Depends(get_database)):
    """Search for faculty mentors using MongoDB regex or semantic search."""
    try:
        if request.search_type == "semantic" and request.query:
            if mentor_retriever is None:
                raise HTTPException(status_code=503, detail="Semantic search unavailable.")
            
            results = mentor_retriever.invoke(request.query)
            # Retrieve ObjectId strings from Chroma metadata
            faculty_ids = []
            for doc in results:
                try:
                    faculty_ids.append(ObjectId(doc.metadata['faculty_id']))
                except InvalidId:
                    continue
            
            cursor = database.faculty.find({"_id": {"$in": faculty_ids}})
            matched_faculty = [parse_mongo_doc(doc) async for doc in cursor]
            return {"mentors": matched_faculty, "search_type": "semantic"}
            
        else:
            # MongoDB Regex Keyword Search
            regex_query = {"$regex": request.query, "$options": "i"}
            cursor = database.faculty.find({
                "$or": [
                    {"name": regex_query},
                    {"expertise": regex_query},
                    {"subjects": regex_query},
                    {"research_areas": regex_query},
                    {"department": regex_query}
                ]
            })
            
            mentors = [parse_mongo_doc(doc) async for doc in cursor]
            return {"mentors": mentors, "search_type": "keyword"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mentor search error: {e}")
        raise HTTPException(status_code=500, detail="Error searching mentors")

@app.post("/api/mentor-request")
async def create_mentor_request(request: MentorshipRequest, database = Depends(get_database)):
    """Create a mentorship request and save to MongoDB."""
    try:
        try:
            faculty_obj_id = ObjectId(request.faculty_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid faculty ID format")

        faculty = await database.faculty.find_one({"_id": faculty_obj_id})
        if not faculty:
            raise HTTPException(status_code=404, detail="Faculty member not found")
        
        request_doc = {
            "student_name": request.student_name,
            "student_email": request.student_email,
            "faculty_id": faculty_obj_id,
            "message": request.message,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat()
        }
        
        result = await database.requests.insert_one(request_doc)
        
        return {
            "success": True,
            "message": "Mentorship request sent successfully",
            "request_id": str(result.inserted_id)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create mentor request error: {e}")
        raise HTTPException(status_code=500, detail="Error creating mentorship request")

@app.get("/api/requests/{email}")
async def get_user_requests(email: str, database = Depends(get_database)):
    """Get mentorship requests for a user utilizing MongoDB Aggregation."""
    try:
        # MongoDB Aggregation to perform a SQL-like JOIN
        pipeline = [
            {"$match": {"student_email": email}},
            {"$lookup": {
                "from": "faculty",
                "localField": "faculty_id",
                "foreignField": "_id",
                "as": "faculty_details"
            }},
            {"$unwind": {
                "path": "$faculty_details",
                "preserveNullAndEmptyArrays": True
            }},
            {"$sort": {"created_at": -1}}
        ]
        
        cursor = database.requests.aggregate(pipeline)
        requests = []
        
        async for doc in cursor:
            # Format the document to match the old SQLite output structure
            formatted_doc = parse_mongo_doc(doc)
            if "faculty_details" in formatted_doc and formatted_doc["faculty_details"]:
                formatted_doc["faculty_name"] = formatted_doc["faculty_details"].get("name")
                formatted_doc["department"] = formatted_doc["faculty_details"].get("department")
                del formatted_doc["faculty_details"]
            requests.append(formatted_doc)
            
        return {"requests": requests}
    except Exception as e:
        logger.error(f"Get user requests error: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving requests")

@app.get("/api/stats")
async def get_stats(database = Depends(get_database)):
    """Get platform statistics using MongoDB counts."""
    try:
        faculty_count = await database.faculty.count_documents({})
        requests_count = await database.requests.count_documents({})
        approved_count = await database.requests.count_documents({"status": "approved"})
        
        return {
            "faculty_count": faculty_count,
            "total_requests": requests_count,
            "approved_requests": approved_count,
            "internships_count": len(INTERNSHIPS_DATA),
            "courses_count": len(COURSES_DATA)
        }
    except Exception as e:
        logger.error(f"Get stats error: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving statistics")

@app.post("/api/interview/start", response_model=InterviewStartResponse)
async def start_interview(request: InterviewStartRequest, database = Depends(get_database)):
    if llm_model is None:
        raise HTTPException(status_code=503, detail="AI service unavailable.")
    
    try:
        # Prompt for the initial question
        prompt = PromptTemplate(
            template="""You are an expert technical interviewer. Generate the FIRST interview question for a candidate.
            Field: {field} | Topic: {topic} | Difficulty: {difficulty}
            
            IMPORTANT: Output ONLY valid JSON containing a single key "question". Start immediately with {{ and end with }}.
            """,
            input_variables=["field", "topic", "difficulty"]
        )
        
        chain = prompt | llm_model | StrOutputParser()
        raw_response = chain.invoke({
            "field": request.field,
            "topic": request.topic,
            "difficulty": request.difficulty
        })
        
        json_str = extract_json_from_text(raw_response)
        import json
        parsed_data = json.loads(json_str)
        question = parsed_data.get("question", f"Could you walk me through your experience with {request.topic}?")
        
        # Save session memory to MongoDB
        session_doc = {
            "user_email": request.user_email,
            "field": request.field,
            "topic": request.topic,
            "difficulty": request.difficulty,
            "status": "in_progress",
            "history": [
                {"role": "interviewer", "content": question}
            ],
            "created_at": datetime.utcnow().isoformat()
        }
        
        result = await database.interviews.insert_one(session_doc)
        
        return InterviewStartResponse(session_id=str(result.inserted_id), question=question)
        
    except Exception as e:
        logger.error(f"Interview start error: {e}")
        raise HTTPException(status_code=500, detail="Failed to start interview.")

@app.post("/api/interview/answer", response_model=InterviewAnswerResponse)
async def submit_interview_answer(request: InterviewAnswerRequest, database = Depends(get_database)):
    try:
        session_id = ObjectId(request.session_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid session format")

    session = await database.interviews.find_one({"_id": session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    try:
        # Format the memory history for the AI Context window
        history_text = ""
        for turn in session["history"]:
            role = turn["role"].capitalize()
            content = turn.get("content", "")
            feedback = turn.get("feedback", "")
            if feedback:
                history_text += f"{role} Feedback: {feedback}\n"
            history_text += f"{role}: {content}\n"
            
        history_text += f"Candidate: {request.answer}\n"
        
        prompt = PromptTemplate(
            template="""You are an expert technical interviewer. Review the conversation history and the candidate's latest answer.
            Field: {field} | Topic: {topic} | Difficulty: {difficulty}
            
            Conversation History:
            {history}
            
            Task:
            1. Provide brief, constructive feedback on their last answer.
            2. Ask the NEXT interview question to continue the technical assessment.
            
            IMPORTANT: Output ONLY valid JSON with two keys: "feedback" and "next_question". Start immediately with {{ and end with }}.
            """,
            input_variables=["field", "topic", "difficulty", "history"]
        )
        
        chain = prompt | llm_model | StrOutputParser()
        raw_response = chain.invoke({
            "field": session["field"],
            "topic": session["topic"],
            "difficulty": session["difficulty"],
            "history": history_text[-3000:] # Keep context manageable to avoid token limits
        })
        
        json_str = extract_json_from_text(raw_response)
        import json
        parsed_data = json.loads(json_str)
        
        feedback = parsed_data.get("feedback", "Noted.")
        next_question = parsed_data.get("next_question", "Let's move to the next topic. What is your greatest strength?")
        
        # Update MongoDB history array
        await database.interviews.update_one(
            {"_id": session_id},
            {"$push": {
                "history": {
                    "$each": [
                        {"role": "candidate", "content": request.answer},
                        {"role": "interviewer", "feedback": feedback, "content": next_question}
                    ]
                }
            }}
        )
        
        return InterviewAnswerResponse(feedback=feedback, next_question=next_question)
        
    except Exception as e:
        logger.error(f"Interview answer error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process answer.")

@app.post("/api/interview/end", response_model=ScorecardResponse)
async def end_interview(request: InterviewEndRequest, database = Depends(get_database)):
    try:
        session_id = ObjectId(request.session_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid session format")

    session = await database.interviews.find_one({"_id": session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    try:
        # Build the full transcript for final evaluation
        history_text = "\n".join([f"{t['role'].capitalize()}: {t.get('content', '')}" for t in session["history"]])
        
        parser = JsonOutputParser(pydantic_object=ScorecardResponse)
        prompt = PromptTemplate(
            template="""You are an expert technical interviewer. The interview is over. Review the full transcript and generate a final scorecard.
            Field: {field} | Topic: {topic}
            
            Transcript:
            {history}
            
            {format_instructions}
            IMPORTANT: Output ONLY valid JSON. Start immediately with {{ and end with }}.
            """,
            input_variables=["field", "topic", "history"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        chain = prompt | llm_model | StrOutputParser()
        raw_response = chain.invoke({
            "field": session["field"],
            "topic": session["topic"],
            "history": history_text[-4000:] 
        })
        
        json_str = extract_json_from_text(raw_response)
        parsed_data = parser.parse(json_str)
        
        # Bulletproof parsing fallback
        scorecard = ScorecardResponse(
            overall_score=int(parsed_data.get("overall_score", 70)),
            strengths=parsed_data.get("strengths", ["Attempted all questions"]),
            weaknesses=parsed_data.get("weaknesses", ["Needs more depth in answers"]),
            detailed_feedback=parsed_data.get("detailed_feedback", "Solid effort. Keep practicing your fundamentals.")
        )
        
        # Update Database
        await database.interviews.update_one(
            {"_id": session_id},
            {"$set": {"status": "completed", "scorecard": scorecard.model_dump()}}
        )
        
        return scorecard
        
    except Exception as e:
        logger.error(f"Interview end error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate scorecard.")

@app.post("/api/outreach/generate", response_model=OutreachResponse)
async def generate_outreach(request: OutreachRequest):
    """Generate highly personalized cold outreach messages."""
    if llm_model is None:
        raise HTTPException(status_code=503, detail="AI service unavailable. Please try again later.")
        
    try:
        parser = JsonOutputParser(pydantic_object=OutreachResponse)
        
        prompt = PromptTemplate(
            template="""You are an expert career coach and executive copywriter. 
            Write three highly personalized cold outreach messages for a job seeker to send to recruiters or hiring managers.
            
            User's Skills & Strengths: {skills}
            Target Role: {role}
            Target Company: {company}
            
            Create 3 variations:
            1. "email_formal": A professional, well-structured cold email highlighting value.
            2. "linkedin_dm": A short, engaging connection request note (under 300 characters).
            3. "email_short": A punchy, direct email focused on a quick chat.
            
            {format_instructions}
            IMPORTANT: Output ONLY valid JSON. Start immediately with {{ and end with }}. Do not use placeholders like [Your Name], use realistic generic names if needed, but focus on the body text.
            """,
            input_variables=['skills', 'role', 'company'],
            partial_variables={'format_instructions': parser.get_format_instructions()}
        )
        
        chain = prompt | llm_model | StrOutputParser()
        raw_response = chain.invoke({
            'skills': request.skills_and_strengths,
            'role': request.target_role,
            'company': request.target_company
        })
        
        json_str = extract_json_from_text(raw_response)
        parsed_data = parser.parse(json_str)
        
        # Bulletproof parsing with fallbacks
        return OutreachResponse(
            email_formal=parsed_data.get('email_formal', "Hi Hiring Team, I noticed the open role and believe my skills are a strong match..."),
            linkedin_dm=parsed_data.get('linkedin_dm', f"Hi! I'm an engineer passionate about {request.target_company}. Let's connect!"),
            email_short=parsed_data.get('email_short', "Hi, I have a strong background in this stack and would love to chat about the role.")
        )
        
    except Exception as e:
        logger.error(f"Outreach generation error: {e}")
        # FALLBACK: Ensure the UI never crashes
        return OutreachResponse(
            email_formal=f"Subject: Experienced candidate for {request.target_role}\n\nHi Hiring Manager,\n\nI am reaching out regarding the {request.target_role} position at {request.target_company}. With my background in {request.skills_and_strengths}, I would bring immediate value to your team. I would love to schedule a brief call to discuss.",
            linkedin_dm=f"Hi! I'm very impressed with the work at {request.target_company}. I'd love to connect and follow your team's updates.",
            email_short=f"Hi,\n\nI noticed {request.target_company} is hiring for a {request.target_role}. Given my experience with {request.skills_and_strengths}, I believe I'd be a great fit. Do you have 5 minutes for a quick chat next week?"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )