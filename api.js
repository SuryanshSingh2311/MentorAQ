// api.js - Frontend API Integration for MentorAQ (Production-Ready)

// Configuration
const API_BASE_URL = 'http://localhost:8000/api';
const API_TIMEOUT = 120000; // 60 seconds for LLM operations

// Utility function to handle API errors with detailed messaging
function handleApiError(error) {
    console.error('API Error:', error);
    
    let errorMessage = 'An unknown error occurred';
    
    if (error.message) {
        errorMessage = error.message;
    } else if (error.detail) {
        errorMessage = error.detail;
    } else if (typeof error === 'string') {
        errorMessage = error;
    }
    
    return {
        success: false,
        error: errorMessage
    };
}

// Utility function for fetch with timeout
async function fetchWithTimeout(url, options = {}, timeout = API_TIMEOUT) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal
        });
        clearTimeout(timeoutId);
        return response;
    } catch (error) {
        clearTimeout(timeoutId);
        if (error.name === 'AbortError') {
            throw new Error('Request timeout - the server is taking too long to respond. Please try again.');
        }
        throw error;
    }
}


// Helper to update ATS score with radial progress animation
function updateAtsScoreDisplay(score) {
    const circle = document.getElementById('atsScoreCircle');
    const scoreText = document.getElementById('atsScore');
    
    // Ensure the score is clamped (0-100)
    const clampedScore = Math.max(0, Math.min(100, score));

    // Determine color based on score
    let color = '#ef4444'; // Red (Danger)
    if (clampedScore >= 75) {
        color = '#48bb78'; // Green (Success)
    } else if (clampedScore >= 50) {
        color = '#fbbf24'; // Yellow (Warning)
    }

    // --- FIX START ---
    
    // Clear any existing interval to prevent overlap issues
    if (scoreText.__animationId) {
        clearInterval(scoreText.__animationId);
    }
    
    // Safely parse current score, defaulting to 0 if content is non-numeric (e.g., "-1028" from prior error)
    let currentScore = 0; 
    // We explicitly set the score to 0 to start the animation cleanly from the bottom.
    // Setting textContent to 0 here ensures the subsequent animation loop works cleanly.
    scoreText.textContent = 0; 
    
    const increment = clampedScore > currentScore ? 1 : -1;
    
    const animateScore = setInterval(() => {
        currentScore += increment;
        
        // Stop condition check
        if ((increment > 0 && currentScore >= clampedScore) || (increment < 0 && currentScore <= clampedScore) || currentScore === clampedScore) {
            currentScore = clampedScore; // Ensure final score is exact
            clearInterval(scoreText.__animationId);
        }
        
        scoreText.textContent = currentScore;
        circle.style.background = `conic-gradient(${color} ${currentScore}%, #eee ${currentScore}%)`;

        if (currentScore === clampedScore) {
            clearInterval(scoreText.__animationId);
        }
        
    }, 20);

    // Store the interval ID on the element for clean termination
    scoreText.__animationId = animateScore; 

    // --- FIX END ---
}

// Helper to show error notifications
function showErrorNotification(message, duration = 5000) {
    const notification = document.createElement('div');
    notification.className = 'error-notification';
    notification.innerHTML = `
        <i class="fas fa-exclamation-circle"></i>
        <span>${message}</span>
        <button onclick="this.parentElement.remove()">&times;</button>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => notification.remove(), 300);
    }, duration);
}

// Helper to show success notifications
function showSuccessNotification(message, duration = 3000) {
    const notification = document.createElement('div');
    notification.className = 'success-notification';
    notification.innerHTML = `
        <i class="fas fa-check-circle"></i>
        <span>${message}</span>
        <button onclick="this.parentElement.remove()">&times;</button>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => notification.remove(), 300);
    }, duration);
}

// =====================================
// API Health Check
// =====================================
async function checkAPIHealth() {
    try {
        const response = await fetchWithTimeout(`http://localhost:8000/health`, {}, 5000);
        const data = await response.json();
        
        console.log('API Health Status:', data);
        
        if (!data.services.llm_model) {
            showErrorNotification('⚠️ AI model not initialized. Resume analysis may not work properly.');
        }
        
        return data;
    } catch (error) {
        console.error('API Health Check Failed:', error);
        showErrorNotification('⚠️ Cannot connect to backend server. Please ensure it is running on port 8000.');
        return null;
    }
}

// =====================================
// Resume Analysis API
// =====================================
async function analyzeResumeAPI(file, jobDescription = '') {
    try {
        if (!file) {
            throw new Error('No file provided');
        }
        
        if (file.size > 10 * 1024 * 1024) { // 10MB limit
            throw new Error('File size exceeds 10MB limit');
        }
        
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            throw new Error('Only PDF files are supported');
        }
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('job_description', jobDescription);

        const response = await fetchWithTimeout(
            `${API_BASE_URL}/analyze-resume`, 
            {
                method: 'POST',
                body: formData
            },
            90000 // 90 seconds for resume analysis
        );

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error: ${response.status} ${response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Resume analysis error:', error);
        return handleApiError(error);
    }
}

async function analyzeResume() {
    const fileInput = document.getElementById('resumeFile');
    const file = fileInput.files[0];
    
    if (!file) {
        showErrorNotification('Please upload a PDF resume first!');
        return;
    }

    const jobDescription = document.getElementById('jobDescription').value.trim();
    const loadingEl = document.getElementById('loadingResume');
    const resultsEl = document.getElementById('resumeResults');
    
    // Reset visuals and show loading
    resultsEl.style.display = 'none';
    loadingEl.style.display = 'flex';
    updateAtsScoreDisplay(0);

    try {
        const data = await analyzeResumeAPI(file, jobDescription);

        if (data.error || data.detail) {
            throw new Error(data.error || data.detail);
        }

        loadingEl.style.display = 'none';
        resultsEl.style.display = 'block';

        // 1. Update ATS Score (with animation)
        updateAtsScoreDisplay(data.ats_score || 0);

        // 2. Update Summary
        document.getElementById('analysisSummary').textContent = 
            data.analysis || 'Analysis completed successfully.';
        
        // 3. Update Lists
        const formatList = (items, targetId) => {
            const listEl = document.getElementById(targetId);
            if (!items || items.length === 0) {
                listEl.innerHTML = '<li>No items to display</li>';
                return;
            }
            listEl.innerHTML = items.map(item => `<li>${item}</li>`).join('');
        };

        formatList(data.strengths || [], 'strengthsList');
        formatList(data.weaknesses || [], 'weaknessesList');

        // 4. Update Keywords
        const keywords = data.keywords || [];
        if (keywords.length > 0) {
            const skillsHtml = keywords.map(skill => 
                `<span class="skill-tag">${skill}</span>`
            ).join('');
            document.getElementById('extractedSkills').innerHTML = skillsHtml;
        } else {
            document.getElementById('extractedSkills').innerHTML = 
                '<p class="text-muted">No keywords extracted</p>';
        }

        // 5. Update Improvements
        const improvementsText = data.improvements || 'No specific improvements identified.';
        document.getElementById('improvements').innerHTML = `
            <div style="white-space: pre-line; line-height: 1.8;">
                ${improvementsText}
            </div>
        `;
        
        // Show success notification
        showSuccessNotification('✅ Resume analysis completed successfully!');
        
        // Scroll to results
        setTimeout(() => {
            resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 300);

    } catch (error) {
        loadingEl.style.display = 'none';
        showErrorNotification(`Resume analysis failed: ${error.message}`);
        console.error('Resume analysis error:', error);
    }
}

// =====================================
// Career Recommendations API
// =====================================
async function getRecommendationsAPI(skills, careerInterest = '') {
    try {
        if (!skills || skills.length === 0) {
            throw new Error('Skills array cannot be empty');
        }
        
        const response = await fetchWithTimeout(`${API_BASE_URL}/recommendations`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                skills: skills,
                career_interest: careerInterest
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Recommendations API error:', error);
        return handleApiError(error);
    }
}

async function getRecommendations() {
    const skillsInput = document.getElementById('userSkills').value.trim();
    
    if (!skillsInput) {
        showErrorNotification('Please enter your skills first!');
        return;
    }

    const skills = skillsInput
        .split(',')
        .map(s => s.trim())
        .filter(s => s.length > 0);
    
    if (skills.length === 0) {
        showErrorNotification('Please enter at least one valid skill!');
        return;
    }
    
    const careerInterest = document.getElementById('careerInterest').value.trim();
    
    const loadingEl = document.getElementById('loadingRecommender');
    const resultsEl = document.getElementById('recommendations');

    loadingEl.style.display = 'flex';
    resultsEl.style.display = 'none';

    try {
        const data = await getRecommendationsAPI(skills, careerInterest);

        if (data.error) {
            throw new Error(data.error);
        }

        loadingEl.style.display = 'none';
        resultsEl.style.display = 'block';

        // Helper to determine skill match styling
        const getSkillTag = (skill, userSkills, matchColor, baseColor) => {
            const isMatched = userSkills.some(us => 
                us.toLowerCase().includes(skill.toLowerCase()) || 
                skill.toLowerCase().includes(us.toLowerCase())
            );
            const bgColor = isMatched ? matchColor : baseColor;
            return `<span class="skill-tag" style="background: ${bgColor};">${skill}</span>`;
        };

        // Display Internships
        const internships = data.internships || [];
        if (internships.length === 0) {
            document.getElementById('internshipsList').innerHTML = 
                '<p class="text-muted">No internships found matching your skills.</p>';
        } else {
            const internshipsHtml = internships.map(internship => {
                const matchColor = internship.match_score >= 70 ? '#48bb78' : 
                                 internship.match_score >= 50 ? '#fbbf24' : '#ef4444';
                const baseColor = '#fcb69f';
                
                return `
                    <div class="internship-card">
                        <div style="display: flex; justify-content: space-between; align-items: start; flex-wrap: wrap; gap: 1rem;">
                            <h3><i class="fas fa-industry"></i> ${internship.title}</h3>
                            <span class="match-badge" style="background: ${matchColor};">
                                ${Math.round(internship.match_score)}% Match
                            </span>
                        </div>
                        <p class="mt-1"><strong>${internship.company}</strong></p>
                        <p style="color: #666;">
                            <i class="fas fa-map-marker-alt"></i> ${internship.location} | 
                            <i class="fas fa-clock"></i> ${internship.duration} | 
                            <i class="fas fa-rupee-sign"></i> ${internship.stipend}
                        </p>
                        <p style="margin: 10px 0;">${internship.description}</p>
                        <p><strong>Required Skills:</strong></p>
                        <div class="skills-container">
                            ${internship.required_skills.map(skill => 
                                getSkillTag(skill, skills, matchColor, baseColor)
                            ).join('')}
                        </div>
                        <a href="${internship.apply_link}" target="_blank" rel="noopener">
                            <button class="btn btn-primary mt-3">🔗 Apply Now</button>
                        </a>
                    </div>
                `;
            }).join('');
            document.getElementById('internshipsList').innerHTML = internshipsHtml;
        }

        // Display Courses
        const courses = data.courses || [];
        if (courses.length === 0) {
            document.getElementById('coursesList').innerHTML = 
                '<p class="text-muted">No courses found matching your skills.</p>';
        } else {
            const coursesHtml = courses.map(course => {
                const matchColor = course.match_score >= 70 ? '#48bb78' : 
                                 course.match_score >= 50 ? '#fbbf24' : '#ef4444';
                const baseColor = '#4ecdc4';
                
                return `
                    <div class="course-card">
                        <div style="display: flex; justify-content: space-between; align-items: start; flex-wrap: wrap; gap: 1rem;">
                            <h3><i class="fas fa-book-reader"></i> ${course.title}</h3>
                            <span class="match-badge" style="background: ${matchColor};">
                                ${Math.round(course.match_score)}% Match
                            </span>
                        </div>
                        <p class="mt-1"><strong>${course.platform}</strong> - ${course.instructor}</p>
                        <p style="color: #666;">
                            <i class="fas fa-clock"></i> ${course.duration} | 
                            <i class="fas fa-chart-bar"></i> ${course.level} | 
                            <i class="fas fa-star"></i> ${course.rating}/5 | 
                            <i class="fas fa-users"></i> ${course.enrolled}
                        </p>
                        <p style="margin: 10px 0;">${course.description}</p>
                        <p><strong>Skills You'll Learn:</strong></p>
                        <div class="skills-container">
                            ${course.skills_covered.map(skill => 
                                getSkillTag(skill, skills, matchColor, baseColor)
                            ).join('')}
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem; flex-wrap: wrap; gap: 1rem;">
                            <p style="font-size: 1.1em; font-weight: 700;">Price: ${course.price}</p>
                            <a href="${course.link}" target="_blank" rel="noopener">
                                <button class="btn btn-primary">🎓 Enroll Now</button>
                            </a>
                        </div>
                    </div>
                `;
            }).join('');
            document.getElementById('coursesList').innerHTML = coursesHtml;
        }

        showSuccessNotification('✅ Recommendations loaded successfully!');
        
        // Scroll to results
        setTimeout(() => {
            resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 300);

    } catch (error) {
        loadingEl.style.display = 'none';
        showErrorNotification(`Failed to get recommendations: ${error.message}`);
        console.error('Recommendations error:', error);
    }
}

// =====================================
// Mentor Search API
// =====================================
async function searchMentorsAPI(query, searchType = 'keyword') {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/mentors/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                search_type: searchType
            })
        }, 30000); // 30 seconds for semantic search

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Mentor search error:', error);
        return handleApiError(error);
    }
}

async function searchMentors(searchType = null) {
    // Get search type from radio buttons if not provided
    if (!searchType) {
        const checkedRadio = document.querySelector('input[name="searchType"]:checked');
        searchType = checkedRadio ? checkedRadio.value : 'keyword';
    }
    
    const query = document.getElementById('mentorSearch').value.trim();
    
    // Show loading indicator
    document.getElementById('mentorsList').innerHTML = `
        <div class="loading active" style="grid-column: 1/-1;">
            <div class="spinner" style="border-top-color: var(--primary-color);"></div>
            <p style="margin-top: 1rem; color: var(--text-dark);">
                Searching for mentors using ${searchType} search...
            </p>
        </div>
    `;

    try {
        const data = await searchMentorsAPI(query, searchType);

        if (data.error) {
            throw new Error(data.error);
        }

        const mentors = data.mentors || [];
        const mentorsListEl = document.getElementById('mentorsList');

        if (mentors.length === 0) {
            mentorsListEl.innerHTML = `
                <div class="card p-4 alert-info-bg" style="grid-column: 1/-1;">
                    <i class="fas fa-info-circle icon-large"></i>
                    <div>
                        <p style="margin: 0 0 0.5rem 0; font-weight: 600;">No mentors found</p>
                        <p style="margin: 0;">
                            No mentors match "${query || 'your search'}" using ${searchType} search. 
                            Try different keywords or switch search modes.
                        </p>
                    </div>
                </div>
            `;
            return;
        }

        const mentorsHtml = mentors.map(mentor => `
            <div class="mentor-card">
                <h2><i class="fas fa-user-graduate"></i> ${mentor.name}</h2>
                <p class="mt-1"><strong>${mentor.title}</strong> - ${mentor.department}</p>
                <p style="color: #666;"><i class="fas fa-envelope"></i> ${mentor.email}</p>
                
                <div style="margin: 15px 0;">
                    <p><strong>💼 Expertise:</strong></p>
                    <div class="skills-container">
                        ${mentor.expertise.split(',').map(skill => 
                            `<span class="skill-tag">${skill.trim()}</span>`
                        ).join('')}
                    </div>
                </div>
                
                <p><strong>📚 Subjects:</strong> ${mentor.subjects}</p>
                <p><strong>🔬 Research Areas:</strong> ${mentor.research_areas}</p>
                <p style="color: var(--text-light); margin: 10px 0;">
                    <strong>ℹ️ Bio:</strong> ${mentor.bio}
                </p>
                <p style="font-weight: 700;">⏰ Available: ${mentor.availability}</p>
                
                <button class="btn btn-primary" 
                    onclick="showMentorshipRequestForm(${mentor.id}, '${mentor.name.replace(/'/g, "\\'")}')" 
                    style="width: 100%; margin-top: 1rem;">
                    📬 Request Mentorship
                </button>
            </div>
        `).join('');

        mentorsListEl.innerHTML = mentorsHtml;
        
        if (query) {
            showSuccessNotification(`✅ Found ${mentors.length} mentor(s) for "${query}"`);
        }

    } catch (error) {
        document.getElementById('mentorsList').innerHTML = `
            <div class="card p-4" style="grid-column: 1/-1; background: #fee; border-left: 5px solid #ef4444;">
                <i class="fas fa-exclamation-triangle" style="color: #ef4444; font-size: 1.5rem;"></i>
                <p style="margin: 0.5rem 0 0 0; color: #991b1b;">
                    <strong>Search Error:</strong> ${error.message}
                </p>
            </div>
        `;
        showErrorNotification(`Mentor search failed: ${error.message}`);
        console.error('Mentor search error:', error);
    }
}

// =====================================
// Mentorship Request API & Modal
// =====================================
async function createMentorshipRequestAPI(studentName, studentEmail, facultyId, message) {
    try {
        if (!studentName || !studentEmail || !message) {
            throw new Error('All fields are required');
        }
        
        // Basic email validation
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(studentEmail)) {
            throw new Error('Please enter a valid email address');
        }
        
        const response = await fetchWithTimeout(`${API_BASE_URL}/mentor-request`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                student_name: studentName,
                student_email: studentEmail,
                faculty_id: facultyId,
                message: message
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Mentorship request error:', error);
        return handleApiError(error);
    }
}

function showMentorshipRequestForm(facultyId, facultyName) {
    // Remove any existing modal first
    closeMentorshipForm();
    
    const formHtml = `
        <div class="modal-overlay" id="mentorshipModal" onclick="if(event.target.id === 'mentorshipModal') closeMentorshipForm()">
            <div class="modal-content">
                <h2>📬 Request Mentorship from ${facultyName}</h2>
                <form id="mentorshipForm" onsubmit="submitMentorshipRequest(event, ${facultyId})">
                    <div class="form-group">
                        <label for="studentName">Your Name*</label>
                        <input type="text" id="studentName" class="form-control" required 
                               placeholder="Enter your full name">
                    </div>
                    <div class="form-group">
                        <label for="studentEmail">Your Email*</label>
                        <input type="email" id="studentEmail" class="form-control" required
                               placeholder="your.email@example.com">
                    </div>
                    <div class="form-group">
                        <label for="mentorMessage">Message*</label>
                        <textarea id="mentorMessage" class="form-control" rows="4" required
                            placeholder="Tell the faculty why you want mentorship, your goals, and what you hope to learn..."></textarea>
                    </div>
                    <div style="display: flex; gap: 1rem; margin-top: 1.5rem;">
                        <button type="submit" class="btn btn-primary" style="flex: 1;">
                            ✅ Send Request
                        </button>
                        <button type="button" class="btn" onclick="closeMentorshipForm()" 
                            style="flex: 1; background: #e2e8f0; color: var(--text-dark);">
                            ❌ Cancel
                        </button>
                    </div>
                </form>
            </div>
        </div>
    `;
    document.body.insertAdjacentHTML('beforeend', formHtml);
}

function closeMentorshipForm() {
    const modal = document.getElementById('mentorshipModal');
    if (modal) modal.remove();
}

async function submitMentorshipRequest(event, facultyId) {
    event.preventDefault();

    const studentName = document.getElementById('studentName').value.trim();
    const studentEmail = document.getElementById('studentEmail').value.trim();
    const message = document.getElementById('mentorMessage').value.trim();
    
    // Disable submit button to prevent double-submission
    const submitBtn = event.target.querySelector('button[type="submit"]');
    const originalBtnText = submitBtn.innerHTML;
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';

    try {
        const data = await createMentorshipRequestAPI(
            studentName, 
            studentEmail, 
            facultyId,
            message
        );

        if (data.error) {
            throw new Error(data.error);
        }

        if (data.success) {
            showSuccessNotification('✅ Request sent successfully! The mentor will contact you soon.');
            closeMentorshipForm();
        }

    } catch (error) {
        showErrorNotification(`Failed to send request: ${error.message}`);
        console.error('Mentorship request error:', error);
        
        // Re-enable button
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalBtnText;
    }
}

// =====================================
// Platform Stats API
// =====================================
async function loadPlatformStats() {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/stats`, {}, 5000);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Update stats on homepage
        const statCards = document.querySelectorAll('#statsContainer .stat-number');
        if (statCards.length >= 4) {
            statCards[0].textContent = data.faculty_count || '5';
            statCards[1].textContent = data.internships_count || '6';
            statCards[2].textContent = data.courses_count || '4';
            statCards[3].textContent = data.total_requests || '0';
        }

    } catch (error) {
        console.error('Error loading stats:', error);
        // Set fallback values
        const statCards = document.querySelectorAll('#statsContainer .stat-number');
        if (statCards.length >= 4) {
            statCards[0].textContent = '5+';
            statCards[1].textContent = '6+';
            statCards[2].textContent = '4+';
            statCards[3].textContent = '0';
        }
    }
}
// =====================================
// Career Roadmap API
// =====================================
async function generateRoadmapAPI(goal, skills, duration) {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/roadmap`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                goal: goal,
                skills: skills,
                duration: parseInt(duration)
            })
        }, 90000); // 90 seconds for LLM generation

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Roadmap API error:', error);
        return handleApiError(error);
    }
}

async function generateRoadmap() {
    const goal = document.getElementById('roadmapGoal').value.trim();
    const skills = document.getElementById('roadmapSkills').value.trim();
    const duration = document.getElementById('roadmapDuration').value;
    
    if (!goal) {
        showErrorNotification('Please enter a goal for your roadmap!');
        return;
    }

    const loadingEl = document.getElementById('loadingRoadmap');
    const resultsEl = document.getElementById('roadmapResults');
    const contentEl = document.getElementById('roadmapContent');

    // Reset visuals
    loadingEl.style.display = 'flex';
    resultsEl.style.display = 'none';

    try {
        const data = await generateRoadmapAPI(goal, skills, duration);

        if (data.error) {
            throw new Error(data.error);
        }

        loadingEl.style.display = 'none';
        resultsEl.style.display = 'block';

        // Update the intro text
        document.getElementById('roadmapIntro').innerHTML = `
            Your tailored roadmap to achieve <strong>"${data.goal}"</strong> in <strong>${data.duration} weeks</strong>.
        `;

        // Render the steps
        const stepsHtml = data.steps.map(step => `
          <div class="card p-4 full-width">   
                     <h3 class="mb-2 text-primary">
                    <i class="fas fa-calendar-week"></i> Week ${step.week}: ${step.title}
                </h3>
                <p style="font-size: 1.05em; line-height: 1.6;">${step.description}</p>
                
                <div class="mt-3">
                    <p style="font-weight: 600; color: var(--text-dark); margin-bottom: 0.5rem;">
                        <i class="fas fa-link"></i> Recommended Resources:
                    </p>
                    <div class="skills-container">
                        ${step.resources.map(res => `<span class="skill-tag" style="background: var(--bg-elev); color: var(--text-dark); border-color: var(--border);">${res}</span>`).join('')}
                    </div>
                </div>
            </div>
        `).join('');

        contentEl.innerHTML = stepsHtml;
        showSuccessNotification('✅ Roadmap generated successfully!');
        
        // Scroll to results cleanly
        setTimeout(() => {
            resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 300);

    } catch (error) {
        loadingEl.style.display = 'none';
        showErrorNotification(`Failed to generate roadmap: ${error.message}`);
    }
}

// Add the missing navigation function for your top navbar
function goToRoadmapOrResume() {
    showSection('roadmap');
}
// =====================================
// Initialize on page load
// =====================================
document.addEventListener('DOMContentLoaded', async function() {
    console.log('MentorAQ Frontend Initialized');
    
    // Check API health
    await checkAPIHealth();
    
    // Load platform stats
    await loadPlatformStats();
    
    // Load initial mentors list
    await searchMentors('keyword');
});
// =====================================
// AI Mock Interview Logic
// =====================================
let currentInterviewSessionId = null;

async function startInterview() {
    const field = document.getElementById('intField').value.trim();
    const topic = document.getElementById('intTopic').value.trim();
    const diff = document.getElementById('intDifficulty').value;
    
    if(!field || !topic) {
        showErrorNotification("Please enter both Field and Topic.");
        return;
    }

    document.getElementById('interviewSetup').style.display = 'none';
    document.getElementById('interviewActive').style.display = 'block';
    
    const chatBox = document.getElementById('interviewChat');
    chatBox.innerHTML = '<p style="color: var(--text-muted);"><em>AI is generating your first question...</em></p>';

    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/interview/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_email: "guest@mentoraq.com", // Replace with real auth if available
                field: field,
                topic: topic,
                difficulty: diff
            })
        });
        
        const data = await response.json();
        currentInterviewSessionId = data.session_id;
        
        chatBox.innerHTML = `
            <div style="margin-bottom: 1rem; border-left: 3px solid var(--primary-color); padding-left: 10px;">
                <strong><i class="fas fa-robot text-primary"></i> Interviewer:</strong>
                <p class="mt-1">${data.question}</p>
            </div>
        `;
    } catch (error) {
        showErrorNotification("Failed to start interview.");
        resetInterview();
    }
}

async function submitAnswer() {
    const answerText = document.getElementById('intAnswerText').value.trim();
    if(!answerText || !currentInterviewSessionId) return;

    const chatBox = document.getElementById('interviewChat');
    const btn = document.getElementById('btnSubmitAnswer');
    
    // Append user answer visually
    chatBox.innerHTML += `
        <div style="margin-bottom: 1.5rem; text-align: right; border-right: 3px solid #64748b; padding-right: 10px;">
            <strong>You:</strong>
            <p class="mt-1">${answerText}</p>
        </div>
        <p id="typingIndicator" style="color: var(--text-muted); font-size: 0.9em;"><em>Interviewer is typing...</em></p>
    `;
    
    document.getElementById('intAnswerText').value = '';
    btn.disabled = true;
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/interview/answer`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: currentInterviewSessionId,
                answer: answerText
            })
        });
        
        const data = await response.json();
        document.getElementById('typingIndicator').remove();
        
        // Append AI Feedback & Next Question
        chatBox.innerHTML += `
            <div style="margin-bottom: 1.5rem; border-left: 3px solid var(--primary-color); padding-left: 10px;">
                <p style="color: var(--warning-color); font-size: 0.9em;" class="mb-2"><i class="fas fa-lightbulb"></i> Feedback: ${data.feedback}</p>
                <strong><i class="fas fa-robot text-primary"></i> Interviewer:</strong>
                <p class="mt-1">${data.next_question}</p>
            </div>
        `;
        chatBox.scrollTop = chatBox.scrollHeight;
    } catch (error) {
        document.getElementById('typingIndicator').innerHTML = "Error analyzing answer.";
    } finally {
        btn.disabled = false;
    }
}

async function endInterview() {
    if(!currentInterviewSessionId) return;

    document.getElementById('interviewActive').style.display = 'none';
    const scorecardDiv = document.getElementById('interviewScorecard');
    scorecardDiv.style.display = 'block';
    
    document.getElementById('scoreDetailed').innerHTML = "<em>Analyzing your performance history...</em>";
    
    

    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/interview/end`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: currentInterviewSessionId })
        }, 90000); // Allow extra time for scorecard generation
        
        const data = await response.json();
        
        document.getElementById('scoreValue').textContent = `${data.overall_score}/100`;
        document.getElementById('scoreDetailed').textContent = data.detailed_feedback;
        
        document.getElementById('scoreStrengths').innerHTML = data.strengths.map(s => `<li>${s}</li>`).join('');
        document.getElementById('scoreWeaknesses').innerHTML = data.weaknesses.map(w => `<li>${w}</li>`).join('');
        
        currentInterviewSessionId = null;
    } catch (error) {
        document.getElementById('scoreDetailed').textContent = "Failed to generate your scorecard.";
    }
}

function resetInterview() {
    currentInterviewSessionId = null;
    document.getElementById('interviewScorecard').style.display = 'none';
    document.getElementById('interviewActive').style.display = 'none';
    document.getElementById('interviewSetup').style.display = 'block';
    document.getElementById('intAnswerText').value = '';
}
// =====================================
// Outreach Copilot Logic
// =====================================
async function generateOutreach() {
    const role = document.getElementById('outreachRole').value.trim();
    const company = document.getElementById('outreachCompany').value.trim();
    const skills = document.getElementById('outreachSkills').value.trim();

    if (!role || !company || !skills) {
        showErrorNotification("Please fill in the target role, company, and your skills.");
        return;
    }

    const loadingEl = document.getElementById('loadingOutreach');
    const resultsEl = document.getElementById('outreachResults');

    loadingEl.style.display = 'flex';
    resultsEl.style.display = 'none';

    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/outreach/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target_role: role,
                target_company: company,
                skills_and_strengths: skills
            })
        });

        const data = await response.json();

        // Populate the UI cards
        document.getElementById('resEmailFormal').textContent = data.email_formal;
        document.getElementById('resLinkedIn').textContent = data.linkedin_dm;
        document.getElementById('resEmailShort').textContent = data.email_short;

        loadingEl.style.display = 'none';
        resultsEl.style.display = 'block';
        showSuccessNotification("Messages generated successfully!");

        // Scroll down
        setTimeout(() => {
            resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 200);

    } catch (error) {
        loadingEl.style.display = 'none';
        showErrorNotification("Failed to generate outreach messages. Please try again.");
    }
}

function copyToClipboard(elementId) {
    const text = document.getElementById(elementId).textContent;
    navigator.clipboard.writeText(text).then(() => {
        showSuccessNotification("Copied to clipboard!");
    }).catch(err => {
        console.error("Failed to copy:", err);
        showErrorNotification("Failed to copy text.");
    });
}

window.generateOutreach = generateOutreach;
window.copyToClipboard = copyToClipboard;

window.startInterview = startInterview;
window.submitAnswer = submitAnswer;
window.endInterview = endInterview;
window.resetInterview = resetInterview;
window.analyzeResume = analyzeResume;
window.getRecommendations = getRecommendations;
window.searchMentors = searchMentors;
window.showMentorshipRequestForm = showMentorshipRequestForm;
window.closeMentorshipForm = closeMentorshipForm;
window.submitMentorshipRequest = submitMentorshipRequest;
window.loadPlatformStats = loadPlatformStats;
window.checkAPIHealth = checkAPIHealth;
window.generateRoadmap = generateRoadmap;
window.goToRoadmapOrResume = goToRoadmapOrResume;