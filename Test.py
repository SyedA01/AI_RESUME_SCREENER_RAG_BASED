from rag_engine import (
    extract_skills,
    extract_resume_profile,
    extract_job_profile,
    calculate_match_score
)

resume_text = """
Technical Skills:
Python, SQL, Pandas, FastAPI, LangChain

Projects:
Resume AI using RAG, ChromaDB and Gemini.
Customer Churn Prediction using Machine Learning.

Education:
B.Tech Artificial Intelligence and Data Science
"""

jd_text = """
Job Role: GenAI Engineer

Required Skills:
Python
LangChain
RAG
FastAPI
Vector Database

Preferred Skills:
Docker
AWS
Kubernetes
"""

resume = extract_resume_profile(resume_text)
job = extract_job_profile(jd_text)

score = calculate_match_score(resume, job)

print("Resume skills:", resume.candidate_skills)
print("Project skills:", resume.project_skills)
print("Detected role:", job.job_role)
print("Required skills:", job.required_skills)
print("Preferred skills:", job.preferred_skills)
print("Match results:", score)
