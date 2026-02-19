# main.py - Production-Ready FastAPI Backend with LangChain Integration

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, ValidationError # Added ValidationError
from typing import List, Dict, Any, Optional
import sqlite3
from datetime import datetime
import tempfile
import os
import json
import re
from io import BytesIO
from contextlib import contextmanager, asynccontextmanager
import logging

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load environment variables with error handling
try:
    from dotenv import load_dotenv
    load_dotenv()
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

# --- Database Setup ---
DB_FILE = os.getenv("DATABASE_PATH", "mentors.db")

@contextmanager
def get_db_conn():
    """Context manager for SQLite connection with thread safety."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize database tables and seed sample data."""
    try:
        with get_db_conn() as conn:
            cur = conn.cursor()
            
            cur.execute('''
            CREATE TABLE IF NOT EXISTS faculty (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                title TEXT,
                department TEXT,
                email TEXT,
                expertise TEXT,
                subjects TEXT,
                research_areas TEXT,
                bio TEXT,
                availability TEXT
            )
            ''')
            
            cur.execute('''
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT NOT NULL,
                student_email TEXT,
                faculty_id INTEGER,
                message TEXT,
                status TEXT DEFAULT 'pending',
                created_at TEXT,
                FOREIGN KEY(faculty_id) REFERENCES faculty(id)
            )
            ''')
            
            conn.commit()
            seed_sample_data(conn)
            logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

def seed_sample_data(conn):
    """Seed initial faculty data if the table is empty."""
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM faculty")
        if cur.fetchone()[0] > 0:
            return
        
        sample_faculty = [
            ("Dr. Aditi Sharma", "Associate Professor", "CSE", "aditi.sharma@example.edu",
             "Machine Learning, NLP, Python, Deep Learning", "ML, Data Structures, AI", 
             "NLP, Knowledge Graphs, Computer Vision", "Works on practical ML for education and industry applications.", "Mon/Wed 3-5pm"),
            ("Prof. Rajiv Verma", "Professor", "Mechanical", "r.verma@example.edu",
             "Thermodynamics, Robotics, MATLAB, Control Systems", "Thermo, Robotics, Mechanics", 
             "Robotics control systems, Automation", "Industry-oriented research in robotics and automation.", "Tue/Thu 10-12pm"),
            ("Dr. Meera Iyer", "Assistant Professor", "Electronics", "meera.iyer@example.edu",
             "Embedded Systems, VLSI, C/C++, IoT", "Circuits, Embedded, Digital Design", 
             "Low-power VLSI, IoT Security", "Embedded systems and IoT devices for smart applications.", "Fri 2-4pm"),
            ("Dr. Ankit Gupta", "Assistant Professor", "CSE", "ankit.gupta@example.edu",
             "Web Development, Full Stack, JavaScript, React", "Web Tech, Databases, Software Engineering", 
             "Cloud Computing, DevOps", "Focuses on modern web technologies and cloud solutions.", "Mon/Fri 1-3pm"),
            ("Prof. Sneha Kapoor", "Professor", "Civil", "sneha.kapoor@example.edu",
             "Structural Engineering, CAD, Project Management", "Structures, Construction, Materials", 
             "Sustainable Construction, Smart Cities", "Works on sustainable infrastructure and smart city projects.", "Wed 11-1pm"),
        ]
        
        cur.executemany('''
        INSERT INTO faculty (name, title, department, email, expertise, subjects, research_areas, bio, availability)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', sample_faculty)
        conn.commit()
        logger.info(f"Seeded {len(sample_faculty)} faculty records")
    except Exception as e:
        logger.error(f"Failed to seed sample data: {e}")

def get_db():
    """Dependency that yields a SQLite connection."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# --- Pydantic Models ---

class RawLLMAnalysis(BaseModel):
    """Model to enforce structured JSON output from LLM."""
    ats_score: int = Field(description="Compatibility score between 0 and 100", ge=0, le=100)
    strengths: List[str] = Field(description="3-5 key strengths from the resume")
    weaknesses: List[str] = Field(description="3-5 areas for improvement")
    keywords: List[str] = Field(description="10-15 important skills and technologies")
    analysis_summary: str = Field(description="Concise analysis summary")

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
    id: int
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
    faculty_id: int
    message: str


def _initialize_llm():
    """Initialize LLM model with robust error handling."""
    global llm_model
    
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        logger.error("HUGGINGFACEHUB_API_TOKEN not found in environment variables")
        raise ValueError("HuggingFace API token is required. Set HUGGINGFACEHUB_API_TOKEN in your .env file")
    
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
        raise

def _initialize_embeddings():
    """Initialize embeddings model with error handling."""
    global embeddings_model
    
    try:
        embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            cache_folder=os.getenv("EMBEDDINGS_CACHE", "./embeddings_cache")
        )
        logger.info("Embeddings model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize embeddings model: {e}")
        raise

def _initialize_mentor_retriever(conn: sqlite3.Connection):
    """Initialize Chroma vector store for mentor search."""
    global mentor_retriever, embeddings_model
    
    if embeddings_model is None:
        raise Exception("Embeddings model must be initialized first")
    
    try:
        logger.info("Initializing Chroma vector store...")
        cur = conn.cursor()
        cur.execute("SELECT * FROM faculty")
        all_faculty = [dict(row) for row in cur.fetchall()]
        
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
                metadata={'faculty_id': faculty['id'], 'name': faculty['name']}
            ))
        
        if docs:
            persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
            db = Chroma.from_documents(
                docs, 
                embeddings_model,
                persist_directory=persist_directory
            )
            mentor_retriever = db.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 5}
            )
            logger.info(f"Chroma initialized with {len(docs)} faculty documents")
        else:
            logger.warning("No faculty data found for Chroma initialization")
            mentor_retriever = None
    except Exception as e:
        logger.error(f"Failed to initialize mentor retriever: {e}")
        mentor_retriever = None

# --- Helper Functions ---

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF with enhanced error handling."""
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
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise HTTPException(status_code=400, detail=f"Error extracting text from PDF: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {tmp_path}: {e}")

def extract_json_from_text(text: str) -> str:
    """Extract JSON object from text, handling various formats."""
    # Try to find JSON between curly braces
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    # If no match, return the original text
    return text

def calculate_skill_match(user_skills: List[str], required_skills: List[str]) -> float:
    """Calculate match percentage between user skills and required skills."""
    if not required_skills:
        return 0.0
    
    user_skills_lower = [s.lower().strip() for s in user_skills]
    required_skills_lower = [s.lower().strip() for s in required_skills]
    
    matches = sum(1 for skill in required_skills_lower 
                  if any(user_skill in skill or skill in user_skill 
                         for user_skill in user_skills_lower))
    
    return (matches / len(required_skills)) * 100

# --- Sample Data ---

INTERNSHIPS_DATA = [
    {"id": 1, "title": "Machine Learning Intern", "company": "TechCorp AI", "location": "Remote", "duration": "3 months", "stipend": "₹15,000/month", "required_skills": ["Python", "Machine Learning", "TensorFlow", "Data Analysis"], "description": "Work on cutting-edge ML models for predictive analytics.", "apply_link": "https://example.com/apply/ml-intern"},
    {"id": 2, "title": "Full Stack Developer Intern", "company": "WebSolutions Inc", "location": "Bangalore", "duration": "6 months", "stipend": "₹20,000/month", "required_skills": ["JavaScript", "React", "Node.js", "MongoDB", "REST API"], "description": "Build scalable web applications using MERN stack.", "apply_link": "https://example.com/apply/fullstack"},
    {"id": 3, "title": "Data Science Intern", "company": "DataMinds Analytics", "location": "Mumbai", "duration": "4 months", "stipend": "₹18,000/month", "required_skills": ["Python", "Pandas", "SQL", "Data Visualization", "Statistics"], "description": "Analyze business data and create insightful dashboards.", "apply_link": "https://example.com/apply/ds-intern"},
    {"id": 4, "title": "Robotics Engineering Intern", "company": "RoboTech Industries", "location": "Pune", "duration": "5 months", "stipend": "₹22,000/month", "required_skills": ["Python", "ROS", "Computer Vision", "Control Systems", "C++"], "description": "Design and program autonomous robots.", "apply_link": "https://example.com/apply/robotics"},
    {"id": 5, "title": "IoT Developer Intern", "company": "SmartHome Solutions", "location": "Hyderabad", "duration": "3 months", "stipend": "₹16,000/month", "required_skills": ["Embedded Systems", "IoT", "Arduino", "MQTT", "C++"], "description": "Develop IoT solutions for smart home automation.", "apply_link": "https://example.com/apply/iot"},
    {"id": 6, "title": "Cloud Computing Intern", "company": "CloudTech Services", "location": "Remote", "duration": "4 months", "stipend": "₹17,000/month", "required_skills": ["AWS", "Docker", "Kubernetes", "Linux", "DevOps"], "description": "Deploy and manage cloud infrastructure.", "apply_link": "https://example.com/apply/cloud"},
]

COURSES_DATA = [
    {"id": 1, "title": "Advanced Machine Learning Specialization", "platform": "Coursera", "instructor": "Andrew Ng", "duration": "4 months", "level": "Advanced", "skills_covered": ["Deep Learning", "Neural Networks", "TensorFlow", "Computer Vision", "NLP"], "rating": 4.8, "enrolled": "150K+", "price": "₹3,999", "description": "Master advanced ML techniques including deep learning and CNNs.", "link": "https://coursera.org/specializations/ml-advanced"},
    {"id": 2, "title": "Full Stack Web Development Bootcamp", "platform": "Udemy", "instructor": "Colt Steele", "duration": "3 months", "level": "Intermediate", "skills_covered": ["HTML", "CSS", "JavaScript", "React", "Node.js", "MongoDB"], "rating": 4.7, "enrolled": "200K+", "price": "₹1,499", "description": "Become a full-stack developer with MERN stack.", "link": "https://udemy.com/course/fullstack-bootcamp"},
    {"id": 3, "title": "Data Science Professional Certificate", "platform": "edX", "instructor": "Harvard University", "duration": "6 months", "level": "Beginner to Advanced", "skills_covered": ["Python", "R", "Statistics", "Data Visualization", "Machine Learning"], "rating": 4.9, "enrolled": "100K+", "price": "₹8,999", "description": "Comprehensive data science program from basics to advanced analytics.", "link": "https://edx.org/professional-certificate/data-science"},
    {"id": 4, "title": "AWS Certified Solutions Architect", "platform": "A Cloud Guru", "instructor": "Ryan Kroonenburg", "duration": "2 months", "level": "Intermediate", "skills_covered": ["AWS", "Cloud Architecture", "EC2", "S3", "VPC", "Lambda"], "rating": 4.6, "enrolled": "80K+", "price": "₹4,499", "description": "Prepare for AWS certification and learn to design scalable cloud solutions.", "link": "https://acloudguru.com/course/aws-architect"},
]

# --- FastAPI Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("Starting application initialization...")
    
    try:
        # Initialize database
        init_db()
        
        # Initialize LangChain components
        _initialize_llm()
        _initialize_embeddings()
        
        # Initialize mentor retriever
        with get_db_conn() as conn:
            _initialize_mentor_retriever(conn)
        
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"FATAL: Application startup failed: {e}")
        raise
    
    yield
    
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
    """API health check and information."""
    return {
        "message": "MentorAQ API - Production Ready",
        "version": "2.0.0",
        "status": "online" if llm_model else "degraded",
        "components": {
            "llm": llm_model is not None,
            "embeddings": embeddings_model is not None,
            "retriever": mentor_retriever is not None
        },
        "endpoints": {
            "resume_analysis": "/api/analyze-resume",
            "recommendations": "/api/recommendations",
            "mentors_search": "/api/mentors/search",
            "mentor_request": "/api/mentor-request",
            "stats": "/api/stats"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy" if llm_model else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": os.path.exists(DB_FILE),
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
        raise HTTPException(
            status_code=503, 
            detail="AI service unavailable. Please try again later."
        )
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        file_content = await file.read()
        resume_text = extract_text_from_pdf(file_content)
        
        if len(resume_text) < 50:
            raise HTTPException(
                status_code=400, 
                detail="Resume appears to be empty or too short"
            )
        
        # Prepare parser and prompt
        parser = JsonOutputParser(pydantic_object=RawLLMAnalysis)
        
        analysis_prompt = PromptTemplate(
            template="""You are an expert ATS (Applicant Tracking System) analyst. Analyze the resume and provide your response in TWO parts:

PART 1 - JSON Analysis (must be valid JSON):
{format_instructions}

PART 2 - Improvements (separate section):
After the JSON, provide 5-7 specific, actionable improvements.

Resume Text:
{resume_text}

Job Description:
{job_description}

Remember: Output ONLY valid JSON for Part 1, then add improvements as plain text after.""",
            input_variables=['resume_text', 'job_description'],
            partial_variables={'format_instructions': parser.get_format_instructions()}
        )
        
        # Create and invoke chain
        chain = analysis_prompt | llm_model | StrOutputParser()
        
        # Truncate inputs to manage context length
        resume_input = resume_text[:4000]
        jd_input = job_description[:1500] if job_description else "No specific job description provided."
        
        raw_response = chain.invoke({
            'resume_text': resume_input,
            'job_description': jd_input
        })
        
        logger.info(f"LLM Response (first 500 chars): {raw_response[:500]}")
        
        # --- START OF PARSING BLOCK ---
        # The logic below is modified for clarity and robustness against LLM output inconsistencies.
        
        # Define a fallback response in case parsing fails entirely
        def get_fallback_response(e: Exception = None):
            if e:
                 logger.error(f"Parsing failed, returning fallback. Error: {e}")
            return ResumeAnalysisResponse(
                ats_score=65,
                analysis="Unable to complete full analysis due to unexpected LLM response format. Please ensure your resume is clearly formatted.",
                keywords=["Python", "JavaScript", "SQL", "Communication", "Problem Solving"],
                improvements="1. Add more specific technical skills\n2. Include quantifiable achievements\n3. Tailor your resume to the job description\n4. Use action verbs to describe your experience\n5. Ensure consistent formatting throughout",
                strengths=["Clear structure", "Relevant experience"],
                weaknesses=["Could include more specific metrics", "Technical skills section needs expansion"]
            )
        
        try:
            # 1. Extract JSON part cleanly from the raw response text
            json_str = extract_json_from_text(raw_response)
            
            # 2. Use the LangChain parser. This often returns a dict if the Pydantic object fails to instantiate.
            raw_parsed_data = parser.parse(json_str) 
            
            # 3. Explicitly validate and convert to Pydantic model for attribute access guarantee
            if isinstance(raw_parsed_data, dict):
                # Manually convert the dictionary to the Pydantic model
                parsed_analysis = RawLLMAnalysis(**raw_parsed_data) 
            elif isinstance(raw_parsed_data, RawLLMAnalysis):
                # If the parser successfully returned the object
                parsed_analysis = raw_parsed_data
            else:
                 # Unexpected type from parser
                 raise TypeError(f"Unexpected data type returned from parser: {type(raw_parsed_data)}")
                
            # 4. Extract improvements (plain text section after JSON)
            # Look for common header keywords
            if "improvements" in raw_response.lower() or "actionable" in raw_response.lower():
                # Split based on common headers used by the LLM (e.g., PART 2 - Improvements)
                parts = re.split(r'(?i)(?:PART\s*2\s*-\s*)?Improvements?|Actionable suggestions?[:\s]*', raw_response, 1)
                improvements_text = parts[-1].strip() if len(parts) > 1 else "Focus on aligning your resume with the job requirements and quantifying achievements."
            else:
                improvements_text = "Focus on adding relevant keywords and quantifying your achievements."
            
        except (Exception, ValidationError) as parse_error:
            # Catch all parsing and validation errors
            return get_fallback_response(parse_error)
        
        # --- END OF PARSING BLOCK ---
        
        # Successful return:
        return ResumeAnalysisResponse(
            ats_score=parsed_analysis.ats_score,
            analysis=parsed_analysis.analysis_summary,
            keywords=parsed_analysis.keywords,
            improvements=improvements_text,
            strengths=parsed_analysis.strengths,
            weaknesses=parsed_analysis.weaknesses
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resume analysis error: {e}")
        # This handles errors outside the parsing step (e.g., PDF reading, connection issues)
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred during resume analysis: {str(e)}"
        )

@app.post("/api/recommendations", response_model=RecommendationsResponse)
async def get_recommendations(request: SkillsRequest):
    """Get personalized internship and course recommendations."""
    
    try:
        user_skills = request.skills
        career_interest = request.career_interest or ""
        
        # Score internships
        scored_internships = []
        for internship in INTERNSHIPS_DATA:
            match_score = calculate_skill_match(user_skills, internship['required_skills'])
            internship_copy = internship.copy()
            internship_copy['match_score'] = round(match_score, 2)
            scored_internships.append(InternshipRecommendation(**internship_copy))
        
        scored_internships.sort(key=lambda x: x.match_score, reverse=True)
        
        # Score courses
        scored_courses = []
        for course in COURSES_DATA:
            match_score = calculate_skill_match(user_skills, course['skills_covered'])
            
            # Boost for career interest match
            if career_interest:
                career_lower = career_interest.lower()
                if (career_lower in course['title'].lower() or 
                    career_lower in ' '.join(course['skills_covered']).lower()):
                    match_score = min(match_score + 20, 100)
            
            course_copy = course.copy()
            course_copy['match_score'] = round(match_score, 2)
            scored_courses.append(CourseRecommendation(**course_copy))
        
        scored_courses.sort(key=lambda x: x.match_score, reverse=True)
        
        return RecommendationsResponse(
            internships=scored_internships[:5],
            courses=scored_courses[:5]
        )
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        raise HTTPException(status_code=500, detail="Error generating recommendations")

@app.post("/api/mentors/search")
async def search_mentors(
    request: MentorSearchRequest, 
    conn: sqlite3.Connection = Depends(get_db)
):
    """Search for faculty mentors using keyword or semantic search."""
    
    try:
        cur = conn.cursor()
        
        if request.search_type == "semantic" and request.query:
            if mentor_retriever is None:
                logger.warning("Semantic search requested but retriever not available")
                raise HTTPException(
                    status_code=503, 
                    detail="Semantic search temporarily unavailable. Try keyword search."
                )
            
            # Semantic search
            results = mentor_retriever.invoke(request.query)
            faculty_ids = [doc.metadata['faculty_id'] for doc in results]
            
            if faculty_ids:
                placeholders = ','.join('?' * len(faculty_ids))
                cur.execute(f"SELECT * FROM faculty WHERE id IN ({placeholders})", faculty_ids)
                matched_faculty = [dict(row) for row in cur.fetchall()]
            else:
                matched_faculty = []
            
            return {"mentors": matched_faculty, "search_type": "semantic"}
        
        else:
            # Keyword search
            keyword_q = f"%{request.query.lower()}%"
            
            cur.execute('''
                SELECT * FROM faculty
                WHERE LOWER(name) LIKE ?
                   OR LOWER(expertise) LIKE ?
                   OR LOWER(subjects) LIKE ?
                   OR LOWER(research_areas) LIKE ?
                   OR LOWER(department) LIKE ?
            ''', (keyword_q, keyword_q, keyword_q, keyword_q, keyword_q))
            
            mentors = [dict(row) for row in cur.fetchall()]
            
            return {"mentors": mentors, "search_type": "keyword"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mentor search error: {e}")
        raise HTTPException(status_code=500, detail="Error searching mentors")

@app.post("/api/mentor-request")
async def create_mentor_request(
    request: MentorshipRequest, 
    conn: sqlite3.Connection = Depends(get_db)
):
    """Create a mentorship request."""
    
    try:
        cur = conn.cursor()
        
        # Verify faculty exists
        cur.execute("SELECT id FROM faculty WHERE id = ?", (request.faculty_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Faculty member not found")
        
        created_at = datetime.utcnow().isoformat()
        
        cur.execute('''
            INSERT INTO requests (student_name, student_email, faculty_id, message, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (request.student_name, request.student_email, request.faculty_id, 
              request.message, created_at))
        
        conn.commit()
        request_id = cur.lastrowid
        
        return {
            "success": True,
            "message": "Mentorship request sent successfully",
            "request_id": request_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create mentor request error: {e}")
        raise HTTPException(status_code=500, detail="Error creating mentorship request")

@app.get("/api/requests/{email}")
async def get_user_requests(email: str, conn: sqlite3.Connection = Depends(get_db)):
    """Get mentorship requests for a user."""
    
    try:
        cur = conn.cursor()
        
        cur.execute("""
            SELECT r.*, f.name as faculty_name, f.department
            FROM requests r
            LEFT JOIN faculty f ON r.faculty_id = f.id
            WHERE r.student_email = ?
            ORDER BY r.created_at DESC
        """, (email,))
        
        requests = [dict(row) for row in cur.fetchall()]
        
        return {"requests": requests}
    except Exception as e:
        logger.error(f"Get user requests error: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving requests")

@app.get("/api/stats")
async def get_stats(conn: sqlite3.Connection = Depends(get_db)):
    """Get platform statistics."""
    
    try:
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) FROM faculty")
        faculty_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM requests")
        requests_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM requests WHERE status='approved'")
        approved_count = cur.fetchone()[0]
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )