import PyPDF2
import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter

class ATSChecker:
    def __init__(self):
        # Change from en_core_web_lg to en_core_web_md (medium) or en_core_web_sm (small)
        self.nlp = spacy.load('en_core_web_lg')  # or 'en_core_web_md' or 'en_core_web_sm'
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return None

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common stop words but keep important ones
        doc = self.nlp(text)
        important_tokens = [token.text for token in doc if not token.is_stop or token.text in ['experience', 'skills', 'education']]
        
        return ' '.join(important_tokens)

    def calculate_similarity(self, resume_text, job_description):
        """Calculate similarity between resume and job description"""
        # Preprocess both texts
        processed_resume = self.preprocess_text(resume_text)
        processed_job = self.preprocess_text(job_description)
        
        # Calculate different similarity metrics
        # 1. Skills-based similarity (higher weight)
        resume_skills = set(self.extract_skills(resume_text))
        job_skills = set(self.extract_skills(job_description))
        skills_score = len(resume_skills.intersection(job_skills)) / len(job_skills) if job_skills else 0
        
        # 2. Overall text similarity (lower weight)
        vectorizer = CountVectorizer().fit_transform([processed_resume, processed_job])
        vectors = vectorizer.toarray()
        text_similarity = cosine_similarity(vectors)[0][1]
        
        # Weighted average (70% skills, 30% text similarity)
        final_score = (skills_score * 0.7 + text_similarity * 0.3) * 100
        
        return final_score

    def extract_skills(self, text):
        """Extract skills from text using NLP"""
        doc = self.nlp(text.lower())
        skills = []
        
        # Custom skill patterns (can be expanded)
        skill_patterns = ['seo', 'sem', 'google analytics', 'social media marketing', 'content marketing', 
                         'email marketing', 'ppc', 'facebook ads', 'google ads', 'instagram marketing',
                         'tiktok marketing', 'marketing automation', 'hubspot', 'mailchimp', 'hootsuite',
                         'buffer', 'canva', 'adobe creative suite', 'photoshop', 'illustrator',
                         'copywriting', 'content strategy', 'a/b testing', 'conversion optimization',
                         'google tag manager', 'wordpress', 'html', 'css', 'marketing strategy',
                         'brand management', 'influencer marketing', 'analytics', 'data visualization',
                         'crm', 'lead generation', 'marketing campaigns', 'digital marketing',
                         'social media marketing', 'google adwords', 'search engine optimization',
                         'offline marketing', 'digital marketing', 'b2b', 'b2c', 'business development',
                         'search engine marketing', 'customer relationship management', 'content writing',
                         'sms marketing', 'online advertising', 'kpi reporting', 'roi reporting',
                         'microsoft os', 'macos', 'ms office', 'database management',
                         'budgeting', 'waterfall', 'agile', 'scrum', 'website testing',
                         'jira', 'asana', 'trello', 'adobe photoshop']
        
        for token in doc:
            if token.text in skill_patterns:
                skills.append(token.text)
            
        return list(set(skills))

    def extract_keywords(self, text):
        """Extract important keywords using NLP"""
        doc = self.nlp(text.lower())
        keywords = []
        
        # Important sections to look for
        sections = ['experience', 'education', 'skills', 'projects', 'achievements']
        
        # Extract noun phrases and named entities
        for chunk in doc.noun_chunks:
            # Check if the root of the chunk is not a stop word
            if not chunk.root.is_stop:
                keywords.append(chunk.text)
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'PERSON']:
                keywords.append(ent.text)
        
        # Add important section headers if found
        for section in sections:
            if section in text.lower():
                keywords.append(section)
                
        return list(set(keywords))

    def analyze_resume(self, resume_path, job_description):
        """Main function to analyze resume against job description"""
        resume_text = self.extract_text_from_pdf(resume_path)
        if not resume_text:
            return None
        
        # Calculate various metrics
        similarity_score = self.calculate_similarity(resume_text, job_description)
        resume_skills = self.extract_skills(resume_text)
        job_skills = self.extract_skills(job_description)
        resume_keywords = self.extract_keywords(resume_text)
        job_keywords = self.extract_keywords(job_description)
        
        # Calculate matches
        matching_skills = set(resume_skills).intersection(set(job_skills))
        missing_skills = set(job_skills) - set(resume_skills)
        keyword_matches = set(resume_keywords).intersection(set(job_keywords))
        
        # Detailed analysis report
        report = {
            'similarity_score': round(similarity_score, 2),
            'matching_skills': list(matching_skills),
            'missing_skills': list(missing_skills),
            'resume_skills': resume_skills,
            'keyword_matches': list(keyword_matches),
            'suggestions': []
        }
        
        # Add suggestions
        if len(missing_skills) > 0:
            report['suggestions'].append(f"Add these missing skills: {', '.join(missing_skills)}")
        
        if similarity_score < 50:
            report['suggestions'].append("Consider adding more relevant experience details and project descriptions")
            report['suggestions'].append("Use more industry-specific terminology from the job description")
            
        return report

def main():
    checker = ATSChecker()
    
    job_description = """
    We are looking for a Digital Marketing Specialist with strong experience in online marketing campaigns.
    Required skills:
    - SEO/SEM
    - Social Media Marketing
    - Content Marketing
    - Google Analytics
    - Email Marketing
    - Marketing Automation
    - Adobe Creative Suite
    - Data Analysis
    
    Responsibilities:
    - Develop and execute digital marketing strategies
    - Create and optimize marketing content
    - Manage social media presence
    - Track and analyze campaign performance
    - Drive website traffic and lead generation
    """
    
    result = checker.analyze_resume("path/to/resume.pdf", job_description)
    if result:
        print("\n=== Detailed ATS Analysis Report ===")
        print(f"\nOverall Match Score: {result['similarity_score']}%")
        
        print("\nMatching Skills:")
        for skill in result['matching_skills']:
            print(f"✓ {skill}")
            
        print("\nMissing Skills (Important to Add):")
        for skill in result['missing_skills']:
            print(f"✗ {skill}")
            
        print("\nRelevant Keywords Found:")
        for keyword in result['keyword_matches']:
            print(f"• {keyword}")
            
        print("\nSuggestions for Improvement:")
        for suggestion in result['suggestions']:
            print(f"- {suggestion}")

if __name__ == "__main__":
    main() 