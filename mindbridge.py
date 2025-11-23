import streamlit as st
# Set page config FIRST before any other Streamlit commands
st.set_page_config(
    page_title="MindBridge - AI Mental Health Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
import pandas as pd
import numpy as np
import json
import datetime
from datetime import timezone
import pytz  # For Malaysia timezone (GMT+8)
import re
import hashlib
import time
import random
import requests

# Malaysia timezone
MALAYSIA_TZ = pytz.timezone('Asia/Kuala_Lumpur')

def get_malaysia_time():
    """Get current time in Malaysia timezone (GMT+8)"""
    return datetime.datetime.now(MALAYSIA_TZ)

def format_malaysia_time(dt=None):
    """Format datetime as Malaysia time string"""
    if dt is None:
        dt = get_malaysia_time()
    elif isinstance(dt, str):
        # Parse ISO string and convert to Malaysia time
        dt = datetime.datetime.fromisoformat(dt.replace('Z', '+00:00'))
        dt = dt.astimezone(MALAYSIA_TZ)
    return dt.strftime('%Y-%m-%d %H:%M:%S %Z')

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Charts will be disabled.")

# Gemini AI-Enhanced Sentiment Analysis
class GeminiSentimentAnalyzer:
    """Uses Google Gemini AI for sophisticated sentiment analysis"""
    
    def __init__(self):
        # Always initialize fallback first (in case API fails)
        self._init_fallback()
        
        # Get API key from Streamlit secrets with better error handling
        self.api_key = None
        try:
            # Try different ways to access secrets
            if hasattr(st, 'secrets'):
                if "GEMINI_API_KEY" in st.secrets:
                    self.api_key = st.secrets["GEMINI_API_KEY"]
                    print(f"✅ Gemini API key found: {self.api_key[:20]}...")
                else:
                    print("❌ GEMINI_API_KEY not found in secrets")
                    print(f"Available secrets: {list(st.secrets.keys())}")
            else:
                print("❌ st.secrets not available")
        except Exception as e:
            print(f"❌ Error accessing secrets: {e}")
            self.api_key = None
        
        if self.api_key and len(self.api_key) > 10:
            self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
            self.use_fallback = False
            print("✅ Gemini AI enabled with model: gemini-1.5-flash")
        else:
            st.warning("⚠️ Gemini AI unavailable. Add GEMINI_API_KEY to Streamlit secrets for smart analysis.")
            self.use_fallback = True
            print("⚠️ Using fallback analysis")
    
    def _init_fallback(self):
        """Simple fallback analyzer if no API key"""
        self.positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
            'happy', 'joy', 'love', 'like', 'enjoy', 'pleased', 'satisfied',
            'hope', 'optimistic', 'confident', 'grateful', 'thankful', 'blessed',
            'better', 'improving', 'positive', 'calm', 'relaxed'
        ]
        
        self.negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'sad', 'angry', 'hate',
            'dislike', 'upset', 'frustrated', 'disappointed', 'worried',
            'anxious', 'anxiety', 'depressed', 'depression', 'lonely', 'hopeless', 
            'worthless', 'stress', 'stressed', 'overwhelmed', 'exhausted', 'nervous',
            'scared', 'fear', 'panic', 'crying', 'tired'
        ]
    
    def analyze_sentiment(self, text, conversation_history=None):
        """Analyze with Gemini AI or fallback to simple method"""
        if self.use_fallback:
            return self._simple_analysis(text)
        
        try:
            result = self._gemini_analysis(text, conversation_history)
            st.sidebar.success("✅ Gemini AI worked!")
            return result
        except Exception as e:
            error_msg = str(e)
            print(f"❌ GEMINI ERROR: {error_msg}")
            st.sidebar.error(f"🚨 Gemini Failed: {error_msg[:200]}")
            return self._simple_analysis(text)
    
    def _gemini_analysis(self, text, conversation_history=None):
        """Use Gemini AI for sophisticated analysis"""
        try:
            # Build context from conversation history
            context = ""
            if conversation_history:
                recent = conversation_history[-5:]  # Last 5 messages
                context = "\n".join([f"{m['role']}: {m['content']}" for m in recent])
            
            prompt = f"""You are a clinical mental health AI analyzer. Analyze this patient message for mental health indicators.

{f"Previous conversation context:\n{context}\n" if context else ""}

Current patient message: "{text}"

Analyze and return ONLY valid JSON (no markdown, no explanation):
{{
    "sentiment_score": <float between -1.0 (very negative) and 1.0 (very positive)>,
    "is_sarcastic": <true or false>,
    "true_emotion": "<actual emotion if sarcastic, or 'none' if not>",
    "depression_indicators": <integer count 0-10>,
    "anxiety_indicators": <integer count 0-10>,
    "crisis_indicators": <integer count 0-5>,
    "risk_level": "<Critical or High or Medium or Low>",
    "emotional_state": "<brief 5-10 word description>",
    "key_concerns": ["<concern1>", "<concern2>"],
    "confidence": <float between 0.0 and 1.0>
}}

CRITICAL: Detect sarcasm ("I'm fine" when struggling), minimization, hidden emotions, and consider conversation context."""

            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 1024
                }
            }
            
            print(f"📡 Calling Gemini API: {self.api_url[:60]}...")
            response = requests.post(self.api_url, json=data, timeout=30)
            print(f"📡 Response status: {response.status_code}")
            
            if response.status_code != 200:
                error_detail = response.text[:500]
                print(f"❌ API Error Response: {error_detail}")
                raise Exception(f"API returned {response.status_code}: {error_detail}")
            
            response.raise_for_status()
            
            result = response.json()
            print(f"📡 Got JSON response")
            
            content = result['candidates'][0]['content']['parts'][0]['text']
            print(f"📡 Raw content: {content[:200]}...")
            
            # Clean up response
            content = content.replace('```json', '').replace('```', '').strip()
            
            # Extract JSON if embedded in text
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group()
            
            print(f"📡 Cleaned content: {content[:200]}...")
            
            analysis = json.loads(content)
            analysis['analysis_timestamp'] = get_malaysia_time().isoformat()
            analysis['ai_model'] = 'gemini-2.0-flash'
            
            print(f"✅ Analysis complete: {analysis.get('risk_level')}")
            
            return analysis
            
        except Exception as e:
            print(f"❌ Exception in _gemini_analysis: {type(e).__name__}: {str(e)}")
            raise
    
    def _simple_analysis(self, text):
        """Fallback simple analysis"""
        words = text.lower().split()
        positive = sum(1 for w in words if w.strip('.,!?') in self.positive_words)
        negative = sum(1 for w in words if w.strip('.,!?') in self.negative_words)
        
        score = 0
        if len(words) > 0:
            score = (positive - negative) / len(words) * 2
            score = max(-1.0, min(1.0, score))
        
        return {
            "sentiment_score": score,
            "is_sarcastic": False,
            "true_emotion": "unknown",
            "depression_indicators": negative,
            "anxiety_indicators": negative,
            "crisis_indicators": 0,
            "risk_level": "Medium" if negative > 2 else "Low",
            "emotional_state": "Basic analysis mode (AI unavailable)",
            "key_concerns": [],
            "confidence": 0.5,
            "analysis_timestamp": get_malaysia_time().isoformat(),
            "ai_model": "simple-fallback"
        }

# Mock EMR Database
class EMRDatabase:
    def __init__(self):
        self.patients = {
            "123456789012": {
                "name": "Ahmad bin Ali",
                "age": 35,
                "gender": "Male",
                "phone": "012-3456789",
                "email": "ahmad.ali@email.com",
                "medical_history": [
                    {"date": "2024-01-15", "diagnosis": "Hypertension", "doctor": "Dr. Lim"},
                    {"date": "2023-08-22", "diagnosis": "Type 2 Diabetes", "doctor": "Dr. Wong"},
                    {"date": "2023-03-10", "diagnosis": "Anxiety Disorder", "doctor": "Dr. Rahman"}
                ],
                "medications": [
                    {"name": "Amlodipine", "dosage": "5mg", "frequency": "Once daily"},
                    {"name": "Metformin", "dosage": "500mg", "frequency": "Twice daily"},
                    {"name": "Lorazepam", "dosage": "0.5mg", "frequency": "As needed"}
                ],
                "allergies": ["Penicillin", "Shellfish"],
                "last_visit": "2024-01-15",
                "mental_health_history": [
                    {"date": "2023-03-10", "condition": "Anxiety Disorder", "severity": "Moderate"}
                ]
            },
            "987654321098": {
                "name": "Siti Nurhaliza",
                "age": 28,
                "gender": "Female",
                "phone": "013-9876543",
                "email": "siti.nur@email.com",
                "medical_history": [
                    {"date": "2024-02-20", "diagnosis": "Migraine", "doctor": "Dr. Tan"},
                    {"date": "2023-11-05", "diagnosis": "Depression", "doctor": "Dr. Ahmad"}
                ],
                "medications": [
                    {"name": "Sumatriptan", "dosage": "50mg", "frequency": "As needed"},
                    {"name": "Sertraline", "dosage": "50mg", "frequency": "Once daily"}
                ],
                "allergies": ["Aspirin"],
                "last_visit": "2024-02-20",
                "mental_health_history": [
                    {"date": "2023-11-05", "condition": "Major Depression", "severity": "Moderate to Severe"}
                ]
            },
            "456789123456": {
                "name": "Raj Kumar",
                "age": 42,
                "gender": "Male",
                "phone": "014-5678901",
                "email": "raj.kumar@email.com",
                "medical_history": [
                    {"date": "2024-03-01", "diagnosis": "Chronic Back Pain", "doctor": "Dr. Lee"},
                    {"date": "2023-12-15", "diagnosis": "Insomnia", "doctor": "Dr. Chong"}
                ],
                "medications": [
                    {"name": "Ibuprofen", "dosage": "400mg", "frequency": "Three times daily"},
                    {"name": "Zolpidem", "dosage": "10mg", "frequency": "Before bedtime"}
                ],
                "allergies": ["None known"],
                "last_visit": "2024-03-01",
                "mental_health_history": [
                    {"date": "2023-12-15", "condition": "Sleep Disorder", "severity": "Mild"}
                ]
            }
        }
    
    def get_patient(self, ic_number):
        return self.patients.get(ic_number)
    
    def add_session_record(self, ic_number, session_data):
        if ic_number in self.patients:
            if 'chat_sessions' not in self.patients[ic_number]:
                self.patients[ic_number]['chat_sessions'] = []
            self.patients[ic_number]['chat_sessions'].append(session_data)

# Mental Health Analysis Engine with Gemini AI
class MentalHealthAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = GeminiSentimentAnalyzer()
        
        # Crisis keywords for safety override
        self.crisis_keywords = [
            'suicide', 'suicidal', 'kill myself', 'end it all', 'die', 'dying',
            'death wish', 'no reason to live', 'better off dead', 'harm myself',
            'hurt myself', 'end my life', 'want to die', 'cant go on'
        ]
    
    def analyze_text(self, text, conversation_history=None):
        """Analyze text using Gemini AI with conversation context"""
        # Get AI analysis
        analysis = self.sentiment_analyzer.analyze_sentiment(text, conversation_history)
        
        # Crisis keyword safety override
        text_lower = text.lower()
        crisis_count = sum(1 for keyword in self.crisis_keywords if keyword in text_lower)
        
        if crisis_count > 0:
            analysis['risk_level'] = "Critical"
            analysis['crisis_indicators'] = max(analysis.get('crisis_indicators', 0), crisis_count)
            if 'key_concerns' not in analysis:
                analysis['key_concerns'] = []
            analysis['key_concerns'].insert(0, "CRISIS KEYWORDS DETECTED - IMMEDIATE INTERVENTION REQUIRED")
        
        return analysis
    
    def generate_recommendations(self, analysis_result, patient_history=None):
        """Generate contextual recommendations based on AI analysis"""
        risk_level = analysis_result.get("risk_level", "Low")
        emotional_state = analysis_result.get("emotional_state", "")
        key_concerns = analysis_result.get("key_concerns", [])
        is_sarcastic = analysis_result.get("is_sarcastic", False)
        
        recommendations = {
            "immediate_action": "",
            "recommendations": [],
            "follow_up": "",
            "additional_notes": []
        }
        
        # Add note if sarcasm detected
        if is_sarcastic:
            true_emotion = analysis_result.get('true_emotion', 'unknown')
            recommendations["additional_notes"].append(
                f"⚠️ Sarcasm/Masking Detected: Patient may be hiding true feelings. True emotion: {true_emotion}"
            )
        
        # Add AI confidence note
        confidence = analysis_result.get('confidence', 0)
        if confidence > 0 and confidence < 0.6:
            recommendations["additional_notes"].append(
                f"ℹ️ AI Confidence: {confidence:.0%} - Consider additional clinical assessment"
            )
        
        # Add key concerns
        if key_concerns:
            concerns_text = ", ".join(key_concerns[:3])
            recommendations["additional_notes"].append(f"🎯 Primary concerns: {concerns_text}")
        
        # Add emotional state
        if emotional_state and "Basic analysis" not in emotional_state:
            recommendations["additional_notes"].append(f"Emotional state: {emotional_state}")
        
        # Risk-based recommendations
        if risk_level == "Critical":
            recommendations.update({
                "immediate_action": "🚨 EMERGENCY - IMMEDIATE INTERVENTION REQUIRED",
                "recommendations": [
                    "Contact emergency services (999) immediately if imminent danger",
                    "Activate crisis response team NOW",
                    "Do NOT leave patient alone",
                    "Immediate psychiatric evaluation required within 1 hour",
                    "Implement safety planning protocol",
                    "Contact patient's emergency contact immediately"
                ],
                "follow_up": "Continuous monitoring - within 1 hour"
            })
        elif risk_level == "High":
            recommendations.update({
                "immediate_action": "⚠️ URGENT - Mental health referral needed within 24-48 hours",
                "recommendations": [
                    "Schedule urgent psychiatrist consultation within 48 hours",
                    "Consider immediate counseling/therapy referral",
                    "Review and adjust current medications if applicable",
                    "Implement daily check-ins (phone or in-person)",
                    "Provide crisis hotline numbers: Befrienders 03-76272929",
                    "Assess support system availability"
                ],
                "follow_up": "Within 24-48 hours, then every 2-3 days"
            })
        elif risk_level == "Medium":
            recommendations.update({
                "immediate_action": "📋 Schedule follow-up appointment within 1-2 weeks",
                "recommendations": [
                    "Consider counseling or therapy referral",
                    "Discuss lifestyle modifications (sleep, exercise, diet)",
                    "Introduce stress management techniques",
                    "Evaluate sleep patterns and quality",
                    "Weekly check-ins via phone or video chat"
                ],
                "follow_up": "Within 1-2 weeks"
            })
        else:  # Low risk
            recommendations.update({
                "immediate_action": "✅ Continue supportive care and monitoring",
                "recommendations": [
                    "Maintain regular check-ups",
                    "Encourage healthy lifestyle habits",
                    "Provide mental health education resources",
                    "Keep communication channels open",
                    "Preventive mental wellness strategies"
                ],
                "follow_up": "Regular scheduled visits"
            })
        
        return recommendations

# Initialize global objects
if 'emr_db' not in st.session_state:
    st.session_state.emr_db = EMRDatabase()
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = MentalHealthAnalyzer()

def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ffebee;
        padding: 1rem;
        border-left: 5px solid #f44336;
        margin: 1rem 0;
    }
    .risk-medium {
        background-color: #fff3e0;
        padding: 1rem;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
    }
    .risk-low {
        background-color: #e8f5e8;
        padding: 1rem;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
    .risk-critical {
        background-color: #fce4ec;
        padding: 1rem;
        border-left: 5px solid #e91e63;
        margin: 1rem 0;
        animation: blink 1s linear infinite;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .chat-message {
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-radius: 15px;
        line-height: 1.6;
        font-size: 1.05rem;
    }
    .user-message {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        margin-left: 3rem;
        border: 1px solid #90caf9;
        box-shadow: 0 2px 8px rgba(33, 150, 243, 0.1);
    }
    .bot-message {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        margin-right: 3rem;
        border: 1px solid #ce93d8;
        box-shadow: 0 2px 8px rgba(156, 39, 176, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🧠 MindBridge Navigation")
    
    # Display current Malaysia time
    st.sidebar.markdown(f"🕐 **{format_malaysia_time()}**")
    st.sidebar.markdown("---")
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_type' not in st.session_state:
        st.session_state.user_type = None
    if 'current_patient' not in st.session_state:
        st.session_state.current_patient = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Navigation menu
    if not st.session_state.authenticated:
        page = st.sidebar.selectbox("Choose Access Type", 
                                  ["🏠 Home", "👤 Patient Login", "👨‍⚕️ Doctor Login"])
    else:
        if st.session_state.user_type == "patient":
            page = st.sidebar.selectbox("Patient Portal", 
                                      ["📋 My Profile", "💬 Mental Health Chat", "📊 My Reports"])
        else:
            page = st.sidebar.selectbox("Doctor Portal", 
                                      ["👥 Patient List", "📊 Analytics Dashboard", "📝 Review Reports"])
        
        if st.sidebar.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.user_type = None
            st.session_state.current_patient = None
            st.session_state.chat_history = []
            st.rerun()
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "👤 Patient Login":
        show_patient_login()
    elif page == "👨‍⚕️ Doctor Login":
        show_doctor_login()
    elif page == "📋 My Profile":
        show_patient_profile()
    elif page == "💬 Mental Health Chat":
        show_chat_interface()
    elif page == "📊 My Reports":
        show_patient_reports()
    elif page == "👥 Patient List":
        show_doctor_patient_list()
    elif page == "📊 Analytics Dashboard":
        show_analytics_dashboard()
    elif page == "📝 Review Reports":
        show_doctor_reports()

def show_home_page():
    st.markdown('<h1 class="main-header">🧠 MindBridge</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">AI-Powered Mental Health Support Platform</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 **For Patients**
        - Secure access with IC number
        - AI-powered mental health chat
        - Personalized health insights
        - Easy appointment booking
        """)
    
    with col2:
        st.markdown("""
        ### 👨‍⚕️ **For Healthcare Providers**
        - Comprehensive patient dashboard
        - AI-generated mental health reports
        - Real-time risk assessment
        - Integrated EMR system
        """)
    
    with col3:
        st.markdown("""
        ### 🔒 **Privacy & Security**
        - GDPR compliant data handling
        - End-to-end encryption
        - Audit trail logging
        - Consent management
        """)
    
    st.markdown("---")
    
    # Demo video placeholder
    st.subheader("🎥 System Demo")
    st.info("This is a prototype demonstration of the MindBridge platform. Use the sidebar to explore patient and doctor interfaces.")
    
    # System statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Registered Patients", "2,847")
    with col2:
        st.metric("👨‍⚕️ Healthcare Providers", "156")
    with col3:
        st.metric("💬 Chat Sessions", "12,394")
    with col4:
        st.metric("🎯 Risk Assessments", "3,421")

def show_patient_login():
    st.title("👤 Patient Login")
    st.write("Enter your Malaysian IC number to access your health records and mental health support.")
    
    with st.form("patient_login"):
        ic_number = st.text_input("IC Number (e.g., 123456789012)", max_chars=12)
        consent_checkbox = st.checkbox("I consent to the retrieval and analysis of my health data for mental health assessment")
        submitted = st.form_submit_button("🔐 Login")
        
        if submitted:
            if len(ic_number) != 12 or not ic_number.isdigit():
                st.error("Please enter a valid 12-digit IC number")
            elif not consent_checkbox:
                st.error("Please provide consent to proceed")
            else:
                patient = st.session_state.emr_db.get_patient(ic_number)
                if patient:
                    st.session_state.authenticated = True
                    st.session_state.user_type = "patient"
                    st.session_state.current_patient = ic_number
                    st.success(f"Welcome back, {patient['name']}!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Patient record not found. Please contact your healthcare provider.")
    
    st.markdown("---")
    st.info("**Demo IC Numbers:**\n- 123456789012 (Ahmad bin Ali)\n- 987654321098 (Siti Nurhaliza)\n- 456789123456 (Raj Kumar)")

def show_doctor_login():
    st.title("👨‍⚕️ Healthcare Provider Login")
    
    with st.form("doctor_login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("🔐 Login")
        
        if submitted:
            # Simple demo authentication
            if username in ["dr.lim", "dr.wong", "dr.ahmad"] and password == "demo123":
                st.session_state.authenticated = True
                st.session_state.user_type = "doctor"
                st.success(f"Welcome, {username.title()}!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    st.markdown("---")
    st.info("**Demo Credentials:**\n- Username: dr.lim, dr.wong, or dr.ahmad\n- Password: demo123")

def show_patient_profile():
    st.title("📋 My Health Profile")
    
    patient_data = st.session_state.emr_db.get_patient(st.session_state.current_patient)
    
    if patient_data:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Personal Information")
            st.write(f"**Name:** {patient_data['name']}")
            st.write(f"**Age:** {patient_data['age']}")
            st.write(f"**Gender:** {patient_data['gender']}")
            st.write(f"**Phone:** {patient_data['phone']}")
            st.write(f"**Email:** {patient_data['email']}")
            st.write(f"**Last Visit:** {patient_data['last_visit']}")
        
        with col2:
            st.subheader("🩺 Medical Summary")
            st.write("**Current Medications:**")
            for med in patient_data['medications']:
                st.write(f"- {med['name']} ({med['dosage']}) - {med['frequency']}")
            
            st.write("**Known Allergies:**")
            for allergy in patient_data['allergies']:
                st.write(f"- {allergy}")
        
        st.subheader("📊 Recent Medical History")
        history_df = pd.DataFrame(patient_data['medical_history'])
        st.dataframe(history_df, use_container_width=True)
        
        st.subheader("🧠 Mental Health History")
        if patient_data['mental_health_history']:
            mh_df = pd.DataFrame(patient_data['mental_health_history'])
            st.dataframe(mh_df, use_container_width=True)
        else:
            st.info("No previous mental health records found.")

def show_chat_interface():
    st.title("💭 Safe Space for Your Thoughts")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 15px; color: white; margin-bottom: 20px;'>
        <h3 style='margin: 0; color: white;'>🤍 You're Not Alone</h3>
        <p style='margin: 5px 0 0 0; opacity: 0.9;'>
            This is a safe, judgment-free space. Take your time, and share whatever feels right. 
            I'm here to listen and support you, one step at a time.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat history with a warmer greeting
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hi there 💙 I'm so glad you're here. This is your space to share whatever's on your mind, at your own pace. There's no pressure - just know that I'm here to listen and support you. How are you feeling today?"}
        ]
    
    # Display chat history with softer styling
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_messages:
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot-message">💙 {message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    # Chat input form (forms auto-clear on submit!)
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Share what's on your mind... 💭", 
                                  placeholder="Take your time. There's no rush, and no judgment here. Share whatever feels right for you today.",
                                  height=120, 
                                  key="message_input")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            send_button = st.form_submit_button("📤 Send Message")
        with col2:
            # Clear chat button outside form since it does different action
            pass
    
    # Handle send button
    if send_button:
        if user_input.strip():
            # Add user message
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            
            # Pass conversation history for AI context
            conversation_history = [
                {"role": msg["role"], "content": msg["content"]} 
                for msg in st.session_state.chat_messages
            ]
            
            # Analyze message with AI
            analysis = st.session_state.analyzer.analyze_text(user_input, conversation_history)
            
            # Generate AI response
            ai_response = generate_ai_response(user_input, analysis)
            st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})
            
            # Save session data
            session_data = {
                "timestamp": get_malaysia_time().isoformat(),
                "messages": st.session_state.chat_messages,
                "analysis": analysis
            }
            st.session_state.emr_db.add_session_record(st.session_state.current_patient, session_data)
            
            st.rerun()
    
    # Clear chat button (outside the form)
    col1, col2, col3 = st.columns([1, 1, 2])
    with col2:
        if st.button("🔄 Clear Chat"):
            st.session_state.chat_messages = [
                {"role": "assistant", "content": "Hello! I'm here to support you today. How are you feeling right now?"}
            ]
            st.rerun()
    
    # Real-time analysis sidebar with Gemini AI insights
    if len(st.session_state.chat_messages) > 1:
        st.sidebar.subheader("📊 Real-time AI Analysis")
        
        # Analyze all user messages
        user_messages = [msg["content"] for msg in st.session_state.chat_messages if msg["role"] == "user"]
        if user_messages:
            combined_text = " ".join(user_messages)
            
            # Pass full conversation history
            conversation_history = [
                {"role": msg["role"], "content": msg["content"]} 
                for msg in st.session_state.chat_messages
            ]
            analysis = st.session_state.analyzer.analyze_text(combined_text, conversation_history)
            
            # Risk level indicator
            risk_level = analysis["risk_level"]
            if risk_level == "Critical":
                st.sidebar.markdown('<div class="risk-critical"><strong>⚠️ CRITICAL RISK DETECTED</strong><br>Immediate intervention required</div>', unsafe_allow_html=True)
            elif risk_level == "High":
                st.sidebar.markdown('<div class="risk-high"><strong>🔴 High Risk</strong><br>Professional support recommended</div>', unsafe_allow_html=True)
            elif risk_level == "Medium":
                st.sidebar.markdown('<div class="risk-medium"><strong>🟡 Medium Risk</strong><br>Monitor closely</div>', unsafe_allow_html=True)
            else:
                st.sidebar.markdown('<div class="risk-low"><strong>🟢 Low Risk</strong><br>Continue support</div>', unsafe_allow_html=True)
            
            # Analysis metrics
            st.sidebar.metric("Sentiment Score", f"{analysis['sentiment_score']:.2f}")
            st.sidebar.metric("Depression Indicators", analysis['depression_indicators'])
            st.sidebar.metric("Anxiety Indicators", analysis['anxiety_indicators'])
            
            # Gemini AI Insights
            if analysis.get('is_sarcastic'):
                st.sidebar.warning(f"⚠️ Sarcasm detected: {analysis.get('true_emotion', 'unknown')}")
            
            emotional_state = analysis.get('emotional_state', '')
            if emotional_state and 'Basic analysis' not in emotional_state:
                st.sidebar.write("**😔 Emotional State:**")
                st.sidebar.write(emotional_state)
            
            confidence = analysis.get('confidence', 0)
            if confidence > 0:
                st.sidebar.metric("🎯 AI Confidence", f"{confidence:.0%}")
            
            key_concerns = analysis.get('key_concerns', [])
            if key_concerns:
                st.sidebar.write("**⚠️ Key Concerns:**")
                for concern in key_concerns:
                    st.sidebar.write(f"  • {concern}")
            
            # Show AI model being used
            ai_model = analysis.get('ai_model', 'unknown')
            if ai_model == 'gemini-2.0-flash':
                st.sidebar.success("✨ Powered by Gemini AI")
            elif ai_model == 'simple-fallback':
                st.sidebar.info("ℹ️ Using basic analysis")
            
            # Debug info (helps troubleshoot)
            with st.sidebar.expander("🔧 Debug Info"):
                st.write(f"AI Model: {ai_model}")
                st.write(f"API Key Present: {'Yes' if st.session_state.analyzer.sentiment_analyzer.api_key else 'No'}")
                st.write(f"Using Fallback: {st.session_state.analyzer.sentiment_analyzer.use_fallback}")
                if st.session_state.analyzer.sentiment_analyzer.api_key:
                    st.write(f"API Key (first 20 chars): {st.session_state.analyzer.sentiment_analyzer.api_key[:20]}...")

def generate_ai_response(user_input, analysis):
    """Generate contextual AI responses based on user input and analysis"""
    
    risk_level = analysis["risk_level"]
    sentiment = analysis["sentiment_score"]
    
    # Crisis response
    if risk_level == "Critical":
        return """I hear you, and I'm really concerned about what you're going through right now. Your life matters, and you deserve support and care. 💙

I know things might feel overwhelming, but please know that you don't have to face this alone. There are people who want to help:

🆘 **If you're in immediate danger:**
• Emergency services: **999** (available 24/7)
• Go to your nearest hospital emergency department

💚 **Someone to talk to right now:**
• **Befrienders Malaysia: 03-7627 2929** (24/7, free, confidential)
• You can also reach out to a trusted friend or family member

I care about your wellbeing, and I want you to get the support you deserve. Will you reach out to one of these resources? You're worth it."""

    # High risk response
    elif risk_level == "High":
        responses = [
            "Thank you for trusting me with what you're feeling. I can hear that you're really struggling right now, and that takes courage to share.",
            "I'm really glad you're here and talking about this. What you're experiencing sounds incredibly difficult.",
            "Your feelings are completely valid, and I want you to know that you're not alone in this."
        ]
        base_response = random.choice(responses)
        return f"""{base_response}

💜 **What might help right now:**
• Talking to a mental health professional can make a real difference - they're trained to help with exactly what you're going through
• The Befrienders helpline (03-7627 2929) offers 24/7 support if you need someone to talk to
• Small acts of self-care - a warm shower, your favorite comfort food, or calling someone you trust
• Remember: these heavy feelings won't last forever, even though they feel overwhelming right now

Would you like to talk more about what's been weighing on you? I'm here to listen, without judgment. 💙"""

    # Medium risk response
    elif risk_level == "Medium":
        if sentiment < -0.2:
            return """I can hear that things feel heavy right now. It's completely okay to not be okay - we all have these moments, and reaching out like you're doing takes real strength. 💙

**Some gentle suggestions that might help:**
• Take a few slow, deep breaths (in through your nose, out through your mouth)
• Step outside for some fresh air, even just for a few minutes
• Talk to someone who makes you feel safe and understood
• Do something small that usually brings you comfort - maybe a cup of tea, your favorite music, or a cozy blanket

Remember, you don't have to tackle everything at once. Just this moment, just this breath.

What do you think would feel helpful right now? I'm here to listen. 🌸"""
        else:
            return """Thank you for opening up and sharing this with me. I'm here to listen and support you through whatever you're experiencing, at your own pace. 💜

**Things that might be helpful to explore:**
• Acknowledge what you're feeling, without judging yourself for it - all feelings are valid
• Think about times you've felt this way before and what helped then
• Consider talking with a friend, family member, or counselor you trust
• Remember that difficult feelings are visitors - they don't stay forever

Is there anything specific that's been on your mind that you'd like to talk through together?"""

    # Low risk/neutral response
    else:
        if sentiment > 0.1:
            return """It's wonderful to hear you're doing okay! 💚 Taking time to check in on your mental health shows real self-awareness and care for yourself.

**Ways to keep nurturing your wellbeing:**
• Move your body in ways that feel good - dancing, walking, stretching
• Stay connected with people who lift you up
• Try mindfulness, meditation, or just quiet moments to yourself
• Make time for things that bring you joy

Even on good days, it's great to talk things through. Is there anything on your mind you'd like to explore? 🌟"""
        else:
            return """Thank you for being here and sharing with me. Whatever you're feeling right now is okay - there's no pressure, no judgment, just a safe space to talk. 💙

**Sometimes it helps to:**
• Put words to what's sitting in your heart or mind
• Explore what you're feeling and where it might be coming from
• Think about small, gentle steps that might bring some ease
• Remember that it's perfectly okay to have mixed feelings or uncertain days

What would feel most supportive for you to talk about right now? I'm here, and I'm listening. 🌸"""

def show_patient_reports():
    st.title("📊 My Mental Health Reports")
    
    patient_data = st.session_state.emr_db.get_patient(st.session_state.current_patient)
    
    if 'chat_sessions' in patient_data and patient_data['chat_sessions']:
        st.subheader("📈 Session History")
        
        for i, session in enumerate(patient_data['chat_sessions']):
            with st.expander(f"Session {i+1} - {session['timestamp'][:10]}"):
                
                # Session analysis
                analysis = session.get('analysis', {})
                risk_level = analysis.get('risk_level', 'Unknown')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Risk Level", risk_level)
                    st.metric("Sentiment Score", f"{analysis.get('sentiment_score', 0):.2f}")
                
                with col2:
                    st.metric("Depression Indicators", analysis.get('depression_indicators', 0))
                    st.metric("Anxiety Indicators", analysis.get('anxiety_indicators', 0))
                
                # Show AI insights if available
                if analysis.get('is_sarcastic'):
                    st.warning(f"⚠️ Note: Sarcasm detected in communication. True emotion: {analysis.get('true_emotion')}")
                
                if analysis.get('key_concerns'):
                    st.write("**Key Concerns Identified:**")
                    for concern in analysis['key_concerns']:
                        st.write(f"• {concern}")
                
                # Recommendations
                recommendations = st.session_state.analyzer.generate_recommendations(analysis)
                st.subheader("💡 Recommendations")
                st.write(f"**Immediate Action:** {recommendations['immediate_action']}")
                st.write("**Suggestions:**")
                for rec in recommendations['recommendations']:
                    st.write(f"- {rec}")
                st.write(f"**Follow-up:** {recommendations['follow_up']}")
                
                if recommendations.get('additional_notes'):
                    st.write("**Additional Notes:**")
                    for note in recommendations['additional_notes']:
                        st.info(note)
                
                # Download report
                if st.button(f"📄 Download Report {i+1}"):
                    report_data = generate_patient_report(session, patient_data)
                    st.download_button(
                        label="Download Report",
                        data=report_data,
                        file_name=f"mental_health_report_{i+1}.txt",
                        mime="text/plain"
                    )
    else:
        st.info("No chat sessions found. Start a conversation in the Mental Health Chat to generate reports.")

def show_doctor_patient_list():
    st.title("👥 Patient Management Dashboard")
    
    # Create patient summary
    patients_summary = []
    for ic, patient in st.session_state.emr_db.patients.items():
        last_session = None
        risk_level = "Not Assessed"
        
        if 'chat_sessions' in patient and patient['chat_sessions']:
            last_session = patient['chat_sessions'][-1]
            risk_level = last_session.get('analysis', {}).get('risk_level', 'Unknown')
        
        patients_summary.append({
            "IC Number": ic,
            "Name": patient['name'],
            "Age": patient['age'],
            "Gender": patient['gender'],
            "Last Visit": patient['last_visit'],
            "Mental Health Risk": risk_level,
            "Chat Sessions": len(patient.get('chat_sessions', []))
        })
    
    df = pd.DataFrame(patients_summary)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.selectbox("Filter by Risk Level", 
                                 ["All", "Critical", "High", "Medium", "Low", "Not Assessed"])
    with col2:
        gender_filter = st.selectbox("Filter by Gender", ["All", "Male", "Female"])
    with col3:
        min_sessions = st.number_input("Min Chat Sessions", min_value=0, value=0)
    
    # Apply filters
    filtered_df = df.copy()
    if risk_filter != "All":
        filtered_df = filtered_df[filtered_df["Mental Health Risk"] == risk_filter]
    if gender_filter != "All":
        filtered_df = filtered_df[filtered_df["Gender"] == gender_filter]
    filtered_df = filtered_df[filtered_df["Chat Sessions"] >= min_sessions]
    
    # Display patient list
    st.subheader(f"📋 Patient List ({len(filtered_df)} patients)")
    
    # Color code based on risk level
    def color_risk_level(val):
        if val == "Critical":
            return 'background-color: #ffcdd2'
        elif val == "High":
            return 'background-color: #ffe0b2'
        elif val == "Medium":
            return 'background-color: #fff9c4'
        elif val == "Low":
            return 'background-color: #c8e6c9'
        else:
            return 'background-color: #f5f5f5'
    
    styled_df = filtered_df.style.applymap(color_risk_level, subset=['Mental Health Risk'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Patient detail view
    st.subheader("🔍 Patient Detail View")
    selected_ic = st.selectbox("Select Patient", filtered_df["IC Number"].tolist())
    
    if selected_ic:
        patient_detail = st.session_state.emr_db.get_patient(selected_ic)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Patient Information:**")
            st.write(f"Name: {patient_detail['name']}")
            st.write(f"Age: {patient_detail['age']}")
            st.write(f"Gender: {patient_detail['gender']}")
            st.write(f"Phone: {patient_detail['phone']}")
            
        with col2:
            st.write("**Medical Summary:**")
            st.write(f"Last Visit: {patient_detail['last_visit']}")
            st.write(f"Active Medications: {len(patient_detail['medications'])}")
            st.write(f"Known Allergies: {len(patient_detail['allergies'])}")
            st.write(f"Chat Sessions: {len(patient_detail.get('chat_sessions', []))}")
        
        # Recent chat sessions
        if 'chat_sessions' in patient_detail and patient_detail['chat_sessions']:
            st.write("**Recent Mental Health Assessment:**")
            latest_session = patient_detail['chat_sessions'][-1]
            analysis = latest_session.get('analysis', {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Risk Level", analysis.get('risk_level', 'Unknown'))
            with col2:
                st.metric("Sentiment", f"{analysis.get('sentiment_score', 0):.2f}")
            with col3:
                st.metric("Depression Signs", analysis.get('depression_indicators', 0))
            with col4:
                st.metric("Anxiety Signs", analysis.get('anxiety_indicators', 0))
            
            # Show AI insights
            if analysis.get('is_sarcastic'):
                st.warning(f"⚠️ Sarcasm detected in latest session: {analysis.get('true_emotion')}")
            
            if analysis.get('key_concerns'):
                st.write("**AI-Identified Concerns:**")
                for concern in analysis['key_concerns']:
                    st.write(f"• {concern}")

def show_analytics_dashboard():
    st.title("📊 Mental Health Analytics Dashboard")
    
    # Generate analytics data
    all_sessions = []
    risk_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0, "Not Assessed": 0}
    
    for ic, patient in st.session_state.emr_db.patients.items():
        if 'chat_sessions' in patient:
            for session in patient['chat_sessions']:
                analysis = session.get('analysis', {})
                all_sessions.append({
                    'date': session['timestamp'][:10],
                    'risk_level': analysis.get('risk_level', 'Unknown'),
                    'sentiment': analysis.get('sentiment_score', 0),
                    'depression_indicators': analysis.get('depression_indicators', 0),
                    'anxiety_indicators': analysis.get('anxiety_indicators', 0)
                })
        else:
            risk_counts["Not Assessed"] += 1
    
    # Count risk levels
    for session in all_sessions:
        risk_level = session['risk_level']
        if risk_level in risk_counts:
            risk_counts[risk_level] += 1
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", len(st.session_state.emr_db.patients))
    with col2:
        st.metric("Total Sessions", len(all_sessions))
    with col3:
        high_risk = risk_counts["Critical"] + risk_counts["High"]
        st.metric("High Risk Patients", high_risk)
    with col4:
        avg_sentiment = np.mean([s['sentiment'] for s in all_sessions]) if all_sessions else 0
        st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
    
    # Charts
    if all_sessions:
        if PLOTLY_AVAILABLE:
            col1, col2 = st.columns(2)
        
            with col1:
                # Risk level distribution
                st.subheader("🎯 Risk Level Distribution")
                risk_df = pd.DataFrame(list(risk_counts.items()), columns=['Risk Level', 'Count'])
                fig_pie = px.pie(risk_df, values='Count', names='Risk Level', 
                               color_discrete_map={
                                   'Critical': '#e91e63',
                                   'High': '#f44336', 
                                   'Medium': '#ff9800',
                                   'Low': '#4caf50',
                                   'Not Assessed': '#9e9e9e'
                               })
                st.plotly_chart(fig_pie, use_container_width=True)
        
            with col2:
                # Sentiment distribution
                st.subheader("😊 Sentiment Score Distribution")
                sentiment_data = [s['sentiment'] for s in all_sessions]
                fig_hist = px.histogram(x=sentiment_data, nbins=20, 
                                      title="Distribution of Sentiment Scores")
                st.plotly_chart(fig_hist, use_container_width=True)
        
            # Time series analysis
            if len(all_sessions) > 1:
                st.subheader("📈 Mental Health Trends Over Time")
                sessions_df = pd.DataFrame(all_sessions)
                sessions_df['date'] = pd.to_datetime(sessions_df['date'])
                daily_sentiment = sessions_df.groupby('date')['sentiment'].mean().reset_index()
            
                fig_line = px.line(daily_sentiment, x='date', y='sentiment',
                                 title="Average Daily Sentiment Score")
                st.plotly_chart(fig_line, use_container_width=True)
        
            # Indicator correlation
            st.subheader("🔗 Mental Health Indicators")
            indicators_df = pd.DataFrame(all_sessions)
        
            col1, col2 = st.columns(2)
            with col1:
                fig_scatter = px.scatter(indicators_df, x='depression_indicators', y='anxiety_indicators',
                                       color='risk_level',
                                       title="Depression vs Anxiety Indicators")
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                # Summary statistics
                st.subheader("📋 Summary Statistics")
                st.write("**Depression Indicators:**")
                st.write(f"- Average: {np.mean([s['depression_indicators'] for s in all_sessions]):.1f}")
                st.write(f"- Max: {max([s['depression_indicators'] for s in all_sessions])}")
                
                st.write("**Anxiety Indicators:**")
                st.write(f"- Average: {np.mean([s['anxiety_indicators'] for s in all_sessions]):.1f}")
                st.write(f"- Max: {max([s['anxiety_indicators'] for s in all_sessions])}")
                
                st.write("**Sentiment Scores:**")
                sentiments = [s['sentiment'] for s in all_sessions]
                st.write(f"- Average: {np.mean(sentiments):.2f}")
                st.write(f"- Range: {min(sentiments):.2f} to {max(sentiments):.2f}")
        else:
            # Fallback to basic charts or tables when Plotly is not available
            st.subheader("📊 Data Summary (Charts unavailable)")
            col1, col2 = st.columns(2)
        
            with col1:
                st.subheader("🎯 Risk Level Distribution")
                risk_df = pd.DataFrame(list(risk_counts.items()), columns=['Risk Level', 'Count'])
                st.dataframe(risk_df)
        
            with col2:
                st.subheader("😊 Sentiment Statistics")
                sentiment_data = [s['sentiment'] for s in all_sessions]
                st.write(f"Average Sentiment: {np.mean(sentiment_data):.2f}")
                st.write(f"Min Sentiment: {min(sentiment_data):.2f}")
                st.write(f"Max Sentiment: {max(sentiment_data):.2f}")
    
    else:
        st.info("No chat session data available for analytics. Patients need to complete chat sessions first.")

def show_doctor_reports():
    st.title("📝 Mental Health Report Review")
    
    # Get all patients with chat sessions
    patients_with_sessions = []
    for ic, patient in st.session_state.emr_db.patients.items():
        if 'chat_sessions' in patient and patient['chat_sessions']:
            patients_with_sessions.append((ic, patient['name']))
    
    if not patients_with_sessions:
        st.info("No patient reports available. Patients need to complete chat sessions first.")
        return
    
    # Patient selection
    selected_patient = st.selectbox("Select Patient", 
                                  options=[ic for ic, name in patients_with_sessions],
                                  format_func=lambda x: next(name for ic, name in patients_with_sessions if ic == x))
    
    if selected_patient:
        patient_data = st.session_state.emr_db.get_patient(selected_patient)
        st.subheader(f"📋 Report for {patient_data['name']}")
        
        # Patient summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Age:** {patient_data['age']}")
            st.write(f"**Gender:** {patient_data['gender']}")
        with col2:
            st.write(f"**Total Sessions:** {len(patient_data['chat_sessions'])}")
            st.write(f"**Last Visit:** {patient_data['last_visit']}")
        with col3:
            latest_analysis = patient_data['chat_sessions'][-1].get('analysis', {})
            risk_level = latest_analysis.get('risk_level', 'Unknown')
            st.write(f"**Current Risk:** {risk_level}")
            st.write(f"**Last Session:** {patient_data['chat_sessions'][-1]['timestamp'][:10]}")
        
        # Session reports
        for i, session in enumerate(patient_data['chat_sessions']):
            with st.expander(f"📅 Session {i+1} - {session['timestamp'][:16]}"):
                
                analysis = session.get('analysis', {})
                messages = session.get('messages', [])
                
                # Analysis summary
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🎯 Risk Assessment")
                    risk_level = analysis.get('risk_level', 'Unknown')
                    
                    if risk_level == "Critical":
                        st.markdown('<div class="risk-critical"><strong>⚠️ CRITICAL RISK</strong></div>', unsafe_allow_html=True)
                    elif risk_level == "High":
                        st.markdown('<div class="risk-high"><strong>🔴 HIGH RISK</strong></div>', unsafe_allow_html=True)
                    elif risk_level == "Medium":
                        st.markdown('<div class="risk-medium"><strong>🟡 MEDIUM RISK</strong></div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="risk-low"><strong>🟢 LOW RISK</strong></div>', unsafe_allow_html=True)
                    
                    st.metric("Sentiment Score", f"{analysis.get('sentiment_score', 0):.2f}")
                    st.metric("Depression Indicators", analysis.get('depression_indicators', 0))
                    st.metric("Anxiety Indicators", analysis.get('anxiety_indicators', 0))
                    if analysis.get('crisis_indicators', 0) > 0:
                        st.error(f"⚠️ Crisis Keywords Detected: {analysis.get('crisis_indicators', 0)}")
                    
                    # AI insights
                    if analysis.get('is_sarcastic'):
                        st.warning(f"⚠️ Sarcasm Detected: {analysis.get('true_emotion')}")
                    
                    confidence = analysis.get('confidence', 0)
                    if confidence > 0:
                        st.info(f"AI Confidence: {confidence:.0%}")
                
                with col2:
                    st.subheader("💡 Clinical Recommendations")
                    recommendations = st.session_state.analyzer.generate_recommendations(analysis, patient_data)
                    
                    st.write(f"**Immediate Action:** {recommendations['immediate_action']}")
                    st.write("**Recommendations:**")
                    for rec in recommendations['recommendations']:
                        st.write(f"• {rec}")
                    st.write(f"**Follow-up Timeline:** {recommendations['follow_up']}")
                    
                    if recommendations.get('additional_notes'):
                        st.write("**Additional Notes:**")
                        for note in recommendations['additional_notes']:
                            st.info(note)
                
                # Show key concerns if available
                key_concerns = analysis.get('key_concerns', [])
                if key_concerns:
                    st.subheader("🎯 AI-Identified Concerns")
                    for concern in key_concerns:
                        st.write(f"• {concern}")
                
                # Chat transcript
                st.subheader("💬 Session Transcript")
                user_messages = [msg['content'] for msg in messages if msg['role'] == 'user']
                if user_messages:
                    transcript_text = "\n\n".join([f"Patient: {msg}" for msg in user_messages])
                    st.text_area("Patient Messages", transcript_text, height=200, disabled=True, key=f"transcript_{selected_patient}_{i}")

                # Doctor notes section
                st.subheader("📝 Clinical Notes")
                
                # Initialize session state for this session's notes if it doesn't exist
                notes_key = f"doctor_notes_{selected_patient}_{i}"
                if notes_key not in st.session_state:
                    # Load existing notes from session data if they exist
                    st.session_state[notes_key] = session.get('doctor_notes', '')
                
                # Display existing saved notes if any
                if session.get('doctor_notes'):
                    st.info(f"**Saved Notes:** {session.get('doctor_notes')}")
                
                doctor_notes = st.text_area(f"Add/Edit clinical notes for session {i+1}:", 
                                          value=st.session_state[notes_key],
                                          key=f"notes_input_{selected_patient}_{i}", 
                                          height=100)
                
                if st.button(f"💾 Save Notes for Session {i+1}", key=f"save_{selected_patient}_{i}"):
                    # Save notes to the session data in EMR
                    if selected_patient in st.session_state.emr_db.patients:
                        if 'chat_sessions' in st.session_state.emr_db.patients[selected_patient]:
                            st.session_state.emr_db.patients[selected_patient]['chat_sessions'][i]['doctor_notes'] = doctor_notes
                            st.session_state[notes_key] = doctor_notes
                            st.success("✅ Clinical notes saved successfully!")
                            st.rerun()
                        else:
                            st.error("Error: Session not found")
                    else:
                        st.error("Error: Patient not found")
                
                # Generate comprehensive report
                if st.button(f"📄 Generate Full Report for Session {i+1}", key=f"report_{selected_patient}_{i}"):
                    # Use saved doctor notes if they exist, otherwise use current input
                    report_notes = session.get('doctor_notes', doctor_notes)
                    full_report = generate_comprehensive_report(session, patient_data, report_notes)
                    st.download_button(
                        label="📥 Download Comprehensive Report",
                        data=full_report,
                        file_name=f"comprehensive_report_{patient_data['name'].replace(' ', '_')}_session_{i+1}.txt",
                        mime="text/plain",
                        key=f"download_{selected_patient}_{i}"
                    )

def generate_patient_report(session, patient_data):
    """Generate a patient-friendly report"""
    
    analysis = session.get('analysis', {})
    timestamp = session.get('timestamp', '')
    
    report = f"""
MINDBRIDGE MENTAL HEALTH REPORT
===============================

Patient: {patient_data['name']}
Date: {timestamp[:10]}
Time: {timestamp[11:16]}

ASSESSMENT SUMMARY
------------------
Risk Level: {analysis.get('risk_level', 'Unknown')}
Sentiment Score: {analysis.get('sentiment_score', 0):.2f}
Depression Indicators: {analysis.get('depression_indicators', 0)}
Anxiety Indicators: {analysis.get('anxiety_indicators', 0)}
AI Confidence: {analysis.get('confidence', 0):.0%}
"""

    if analysis.get('is_sarcastic'):
        report += f"\nNote: Communication style detected - True emotion: {analysis.get('true_emotion')}\n"
    
    if analysis.get('emotional_state'):
        report += f"Emotional State: {analysis.get('emotional_state')}\n"
    
    if analysis.get('key_concerns'):
        report += "\nKey Concerns Identified:\n"
        for concern in analysis['key_concerns']:
            report += f"- {concern}\n"
    
    report += """
RECOMMENDATIONS
---------------
"""
    
    recommendations = st.session_state.analyzer.generate_recommendations(analysis)
    report += f"Immediate Action: {recommendations['immediate_action']}\n\n"
    report += "Suggested Next Steps:\n"
    for i, rec in enumerate(recommendations['recommendations'], 1):
        report += f"{i}. {rec}\n"
    
    report += f"\nFollow-up Timeline: {recommendations['follow_up']}\n"
    
    if recommendations.get('additional_notes'):
        report += "\nAdditional Notes:\n"
        for note in recommendations['additional_notes']:
            report += f"- {note}\n"
    
    report += """
IMPORTANT NOTES
---------------
- This report is generated by AI and should be reviewed by a healthcare professional
- If you're experiencing a mental health crisis, contact emergency services (999) immediately
- For ongoing support, contact Befrienders: 03-76272929
- Regular follow-up with your healthcare provider is recommended

PRIVACY NOTICE
--------------
This report contains confidential medical information. Keep it secure and only share with authorized healthcare providers.

Generated by MindBridge AI Mental Health Platform
Powered by Gemini AI
"""
    
    return report

def generate_comprehensive_report(session, patient_data, doctor_notes=""):
    """Generate a comprehensive clinical report"""
    
    analysis = session.get('analysis', {})
    messages = session.get('messages', [])
    timestamp = session.get('timestamp', '')
    
    report = f"""
COMPREHENSIVE MENTAL HEALTH ASSESSMENT REPORT
============================================

PATIENT INFORMATION
-------------------
Name: {patient_data['name']}
Age: {patient_data['age']}
Gender: {patient_data['gender']}
Assessment Date: {timestamp[:10]}
Assessment Time: {timestamp[11:16]}

MEDICAL HISTORY SUMMARY
-----------------------
Last Medical Visit: {patient_data['last_visit']}

Current Medications:
"""
    
    for med in patient_data['medications']:
        report += f"- {med['name']} {med['dosage']} ({med['frequency']})\n"
    
    report += f"\nKnown Allergies: {', '.join(patient_data['allergies'])}\n"
    
    if patient_data['mental_health_history']:
        report += "\nPrevious Mental Health History:\n"
        for mh in patient_data['mental_health_history']:
            report += f"- {mh['date']}: {mh['condition']} ({mh['severity']})\n"
    
    report += f"""

AI ANALYSIS RESULTS (Powered by Gemini AI)
-------------------------------------------
Overall Risk Assessment: {analysis.get('risk_level', 'Unknown')}
Sentiment Analysis Score: {analysis.get('sentiment_score', 0):.3f}
Depression Risk Indicators: {analysis.get('depression_indicators', 0)}
Anxiety Risk Indicators: {analysis.get('anxiety_indicators', 0)}
Crisis Risk Indicators: {analysis.get('crisis_indicators', 0)}
AI Confidence Level: {analysis.get('confidence', 0):.0%}
"""

    if analysis.get('is_sarcastic'):
        report += f"\n⚠️ SARCASM DETECTED: Patient may be masking true emotions\n"
        report += f"True Emotional State: {analysis.get('true_emotion', 'Unknown')}\n"
    
    if analysis.get('emotional_state'):
        report += f"\nEmotional State Assessment: {analysis.get('emotional_state')}\n"
    
    if analysis.get('key_concerns'):
        report += "\nAI-Identified Key Concerns:\n"
        for concern in analysis['key_concerns']:
            report += f"• {concern}\n"
    
    report += """

RISK INTERPRETATION
-------------------
"""
    
    risk_level = analysis.get('risk_level', 'Unknown')
    if risk_level == "Critical":
        report += "CRITICAL: Immediate psychiatric intervention required. Patient may be at risk of self-harm.\n"
    elif risk_level == "High":
        report += "HIGH: Significant mental health concerns detected. Professional evaluation recommended within 1 week.\n"
    elif risk_level == "Medium":
        report += "MEDIUM: Moderate mental health indicators present. Monitoring and support recommended.\n"
    else:
        report += "LOW: Minimal mental health risk indicators detected. Continue routine care.\n"
    
    # Clinical recommendations
    recommendations = st.session_state.analyzer.generate_recommendations(analysis, patient_data)
    
    report += f"""

CLINICAL RECOMMENDATIONS
------------------------
Immediate Action Required: {recommendations['immediate_action']}

Recommended Interventions:
"""
    
    for i, rec in enumerate(recommendations['recommendations'], 1):
        report += f"{i}. {rec}\n"
    
    report += f"\nFollow-up Timeline: {recommendations['follow_up']}\n"
    
    if recommendations.get('additional_notes'):
        report += "\nAdditional Clinical Notes:\n"
        for note in recommendations['additional_notes']:
            report += f"• {note}\n"
    
    # Patient communication analysis
    user_messages = [msg['content'] for msg in messages if msg['role'] == 'user']
    if user_messages:
        report += f"""

COMMUNICATION ANALYSIS
----------------------
Total Patient Messages: {len(user_messages)}
Average Message Length: {np.mean([len(msg.split()) for msg in user_messages]):.1f} words

Key Themes Identified:
"""
        
        # Simple keyword analysis for themes
        all_text = " ".join(user_messages).lower()
        themes = []
        
        if any(word in all_text for word in ['sad', 'depressed', 'down', 'hopeless']):
            themes.append("Depressive symptoms")
        if any(word in all_text for word in ['anxious', 'worried', 'nervous', 'panic']):
            themes.append("Anxiety symptoms")
        if any(word in all_text for word in ['sleep', 'tired', 'fatigue', 'insomnia']):
            themes.append("Sleep disturbances")
        if any(word in all_text for word in ['work', 'job', 'career', 'stress']):
            themes.append("Work-related stress")
        if any(word in all_text for word in ['family', 'relationship', 'partner']):
            themes.append("Relationship concerns")
        
        if themes:
            for theme in themes:
                report += f"- {theme}\n"
        else:
            report += "- General mental health discussion\n"
    
    # Doctor's clinical notes
    if doctor_notes.strip():
        report += f"""

CLINICAL NOTES
--------------
{doctor_notes}
"""
    
    report += f"""

TECHNICAL DETAILS
-----------------
Analysis Engine: MindBridge AI v2.0 with Gemini Pro
Assessment Method: Natural Language Processing + AI Sentiment Analysis
AI Model: {analysis.get('ai_model', 'Unknown')}
Confidence Level: {analysis.get('confidence', 0):.0%}
Data Quality: {'Good' if len(user_messages) > 2 else 'Limited'}

DISCLAIMER
----------
This report is generated using AI technology and should be used as a clinical decision support tool only. 
All recommendations should be reviewed and validated by qualified mental health professionals.
The AI analysis is based on text communication patterns and may not capture all relevant clinical factors.

CONFIDENTIALITY NOTICE
----------------------
This document contains privileged and confidential information intended solely for authorized healthcare providers.
Distribution should be limited to personnel directly involved in patient care.
Ensure compliance with local privacy and data protection regulations.

Report Generated: {get_malaysia_time().strftime('%Y-%m-%d %H:%M:%S')}
Generated by: MindBridge AI Mental Health Platform v2.0
Powered by: Google Gemini AI
"""
    
    return report

if __name__ == "__main__":
    main()
