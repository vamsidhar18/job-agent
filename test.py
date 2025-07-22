from scorer.job_agent import JobScoringAgent

agent = JobScoringAgent()

job = {
    "title": "Software Engineer I",
    "company": "Tesla",
    "description": "We are looking for a backend developer with experience in Python, AWS, and microservices. H1B candidates welcome."
}

print("🔍 Scoring job...")
print(agent.score(job))
print("👍")  # 👍 means the job is a good match for the agent