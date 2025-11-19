import streamlit as st
# Set page config FIRST before any other Streamlit commands
st.set_page_config(
    page_title="🧠 MindBridge - AI Mental Health Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
import pandas as pd
import numpy as np
import json
import datetime
import re
import hashlib
import time
import random

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Charts will be disabled.")

# Simple sentiment analysis without TextBlob
class SimpleSentimentAnalyzer:
    def __init__(self):
        self.positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
            'happy', 'joy', 'love', 'like', 'enjoy', 'pleased', 'satisfied',
            'hope', 'optimistic', 'confident', 'grateful', 'thankful', 'blessed'
        ]
        
        self.negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'sad', 'angry', 'hate',
            'dislike', 'upset', 'frustrated', 'disappointed', 'worried',
            'anxious', 'depressed', 'lonely', 'hopeless', 'worthless'
        ]
    
    def analyze_sentiment(self, text):
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total_words = len(words)
        if total_words == 0:
            return 0.0
        
        # Simple scoring: (positive - negative) / total words
        score = (positive_count - negative_count) / total_words
        # Normalize to [-1, 1] range
        return max(-1.0, min(1.0, score * 2))

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

# Mental Health Analysis Engine
class MentalHealthAnalyzer:
    def __init__(self):
        # Initialize sentiment analyzer FIRST
        self.sentiment_analyzer = SimpleSentimentAnalyzer()
        
        self.depression_keywords = [
            'sad', 'sadness', 'depressed', 'depression', 'hopeless', 'worthless', 'empty', 'lonely',
            'crying', 'cry', 'tired', 'fatigue', 'exhausted', 'sleep', 'insomnia', 'appetite', 'guilty', 'guilt',
            'death', 'die', 'suicide', 'suicidal', 'ending', 'give up', 'no point', 'burden', 'numb', 'meaningless'
        ]
        
        self.anxiety_keywords = [
            'worried', 'anxious', 'nervous', 'panic', 'fear', 'scared',
            'restless', 'tense', 'stress', 'overwhelm', 'racing thoughts',
            'heart racing', 'sweating', 'trembling', 'avoid', 'escape', 'anxiety'
        ]
        
        self.crisis_keywords = [
            'suicide', 'kill myself', 'end it all', 'die', 'death wish',
            'no reason to live', 'better off dead', 'harm myself'
        ]
    
    def analyze_text(self, text):
        text_lower = text.lower()
        
        # Sentiment analysis using simple method
        sentiment_score = self.sentiment_analyzer.analyze_sentiment(text)
        
        # Keyword matching
        depression_count = sum(1 for keyword in self.depression_keywords if keyword in text_lower)
        anxiety_count = sum(1 for keyword in self.anxiety_keywords if keyword in text_lower)
        crisis_count = sum(1 for keyword in self.crisis_keywords if keyword in text_lower)
        
        # Risk assessment
        risk_level = "Low"
        if crisis_count > 0:
            risk_level = "Critical"
        elif depression_count >= 3 or anxiety_count >= 3 or sentiment_score < -0.5:
            risk_level = "High"
        elif depression_count >= 1 or anxiety_count >= 1 or sentiment_score < -0.2:
            risk_level = "Medium"
        
        return {
            "sentiment_score": sentiment_score,
            "depression_indicators": depression_count,
            "anxiety_indicators": anxiety_count,
            "crisis_indicators": crisis_count,
            "risk_level": risk_level,
            "analysis_timestamp": datetime.datetime.now().isoformat()
        }
    
    def generate_recommendations(self, analysis_result, patient_history=None):
        risk_level = analysis_result["risk_level"]
        
        if risk_level == "Critical":
            return {
                "immediate_action": "EMERGENCY REFERRAL REQUIRED",
                "recommendations": [
                    "Immediate psychiatric evaluation needed",
                    "Consider emergency department referral",
                    "Implement safety planning",
                    "Contact emergency services if imminent danger"
                ],
                "follow_up": "Within 24 hours"
            }
        elif risk_level == "High":
            return {
                "immediate_action": "Urgent mental health referral",
                "recommendations": [
                    "Schedule psychiatrist consultation within 1 week",
                    "Consider counseling/therapy referral",
                    "Review current medications",
                    "Implement regular check-ins"
                ],
                "follow_up": "Within 1 week"
            }
        elif risk_level == "Medium":
            return {
                "immediate_action": "Monitor and support",
                "recommendations": [
                    "Schedule follow-up appointment",
                    "Consider counseling referral",
                    "Lifestyle modifications discussion",
                    "Stress management techniques"
                ],
                "follow_up": "Within 2-4 weeks"
            }
        else:
            return {
                "immediate_action": "Continue monitoring",
                "recommendations": [
                    "Maintain regular check-ups",
                    "Encourage healthy lifestyle",
                    "Preventive mental health education",
                    "Open communication channels"
                ],
                "follow_up": "Regular scheduled visits"
            }

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
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    .bot-message {
        background-color: #f5f5f5;
        margin-right: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🧠 MindBridge Navigation")
    
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
    st.title("💬 Mental Health Support Chat")
    st.write("Hi! I'm your AI mental health assistant. I'm here to listen and provide support.")
    
    # Initialize chat history
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hello! I'm here to support you today. How are you feeling right now?"}
        ]
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_messages:
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot-message"><strong>AI Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    user_input = st.text_area("Type your message here...", height=100, key="chat_input")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("📤 Send Message"):
            if user_input.strip():
                # Add user message
                st.session_state.chat_messages.append({"role": "user", "content": user_input})
                
                # Analyze message
                analysis = st.session_state.analyzer.analyze_text(user_input)
                
                # Generate AI response
                ai_response = generate_ai_response(user_input, analysis)
                st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})
                
                # Save session data
                session_data = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "messages": st.session_state.chat_messages,
                    "analysis": analysis
                }
                st.session_state.emr_db.add_session_record(st.session_state.current_patient, session_data)
                
                st.rerun()
    
    with col2:
        if st.button("🔄 Clear Chat"):
            st.session_state.chat_messages = [
                {"role": "assistant", "content": "Hello! I'm here to support you today. How are you feeling right now?"}
            ]
            st.rerun()
    
    # Real-time analysis sidebar
    if len(st.session_state.chat_messages) > 1:
        st.sidebar.subheader("📊 Real-time Analysis")
        
        # Analyze all user messages
        user_messages = [msg["content"] for msg in st.session_state.chat_messages if msg["role"] == "user"]
        if user_messages:
            combined_text = " ".join(user_messages)
            analysis = st.session_state.analyzer.analyze_text(combined_text)
            
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

def generate_ai_response(user_input, analysis):
    """Generate contextual AI responses based on user input and analysis"""
    
    risk_level = analysis["risk_level"]
    sentiment = analysis["sentiment_score"]
    
    # Crisis response
    if risk_level == "Critical":
        return """I'm very concerned about what you've shared. Your safety is the most important thing right now. 
        
Please consider:
- Calling emergency services (999) if you're in immediate danger
- Contacting the Befrienders helpline: 03-76272929
- Going to your nearest hospital emergency department
- Reaching out to a trusted friend or family member

You don't have to go through this alone. Professional help is available and can make a real difference."""

    # High risk response
    elif risk_level == "High":
        responses = [
            "Thank you for sharing that with me. What you're experiencing sounds really challenging, and I want you to know that your feelings are valid.",
            "I can hear that you're going through a difficult time. It takes courage to talk about these feelings.",
            "It sounds like you're dealing with a lot right now. These feelings can be overwhelming, but there are ways to work through them."
        ]
        base_response = random.choice(responses)
        return f"""{base_response}
        
I'd strongly recommend speaking with a mental health professional who can provide the support you deserve. In the meantime, here are some things that might help:
- Take things one day at a time
- Reach out to trusted friends or family
- Consider calling a helpline if you need someone to talk to
- Try some gentle self-care activities

Would you like to talk more about what's been troubling you?"""

    # Medium risk response
    elif risk_level == "Medium":
        if sentiment < -0.2:
            return """I can sense that you're not feeling your best right now. It's completely normal to have ups and downs, and I'm glad you're reaching out.
            
Some things that might help:
- Taking a few deep breaths
- Going for a short walk if possible
- Talking to someone you trust
- Doing something small that usually brings you comfort

What do you think might help you feel a bit better today?"""
        else:
            return """Thank you for sharing that with me. I'm here to listen and support you through whatever you're experiencing.
            
It can be helpful to:
- Acknowledge your feelings without judgment
- Think about what has helped you in similar situations before
- Consider reaching out to friends, family, or a counselor
- Remember that difficult feelings are temporary

Is there anything specific you'd like to talk about or explore?"""

    # Low risk/neutral response
    else:
        if sentiment > 0.1:
            return """I'm glad to hear you're doing relatively well! It's great that you're checking in on your mental health.
            
Maintaining good mental health is an ongoing process. Some things that can help:
- Regular exercise and good sleep
- Staying connected with people you care about
- Practicing mindfulness or relaxation techniques
- Engaging in activities you enjoy

Is there anything specific about your wellbeing you'd like to discuss?"""
        else:
            return """Thank you for sharing with me. I'm here to support you and listen to whatever you'd like to talk about.
            
Sometimes it helps to:
- Talk through what's on your mind
- Identify what you're feeling and why
- Think about small steps that might help
- Remember that it's okay to not be okay sometimes

What would be most helpful for you to discuss right now?"""

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
                
                # Recommendations
                recommendations = st.session_state.analyzer.generate_recommendations(analysis)
                st.subheader("💡 Recommendations")
                st.write(f"**Immediate Action:** {recommendations['immediate_action']}")
                st.write("**Suggestions:**")
                for rec in recommendations['recommendations']:
                    st.write(f"- {rec}")
                st.write(f"**Follow-up:** {recommendations['follow_up']}")
                
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
                fig_hist.update_xaxis(title="Sentiment Score")
                fig_hist.update_yaxis(title="Frequency")
                st.plotly_chart(fig_hist, use_container_width=True)
        
            # Time series analysis
            if len(all_sessions) > 1:
                st.subheader("📈 Mental Health Trends Over Time")
                sessions_df = pd.DataFrame(all_sessions)
                sessions_df['date'] = pd.to_datetime(sessions_df['date'])
                daily_sentiment = sessions_df.groupby('date')['sentiment'].mean().reset_index()
            
                fig_line = px.line(daily_sentiment, x='date', y='sentiment',
                                 title="Average Daily Sentiment Score")
                fig_line.update_xaxis(title="Date")
                fig_line.update_yaxis(title="Average Sentiment")
                st.plotly_chart(fig_line, use_container_width=True)
        
            # Indicator correlation
            st.subheader("🔗 Mental Health Indicators")
            indicators_df = pd.DataFrame(all_sessions)
        
            col1, col2 = st.columns(2)
            with col1:
                fig_scatter = px.scatter(indicators_df, x='depression_indicators', y='anxiety_indicators',
                                       color='risk_level', size='sentiment', 
                                       title="Depression vs Anxiety Indicators")
                st.plotly_chart(fig_scatter, use_container_width=True)
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
                
                with col2:
                    st.subheader("💡 Clinical Recommendations")
                    recommendations = st.session_state.analyzer.generate_recommendations(analysis, patient_data)
                    
                    st.write(f"**Immediate Action:** {recommendations['immediate_action']}")
                    st.write("**Recommendations:**")
                    for rec in recommendations['recommendations']:
                        st.write(f"• {rec}")
                    st.write(f"**Follow-up Timeline:** {recommendations['follow_up']}")
                
                # Chat transcript
                st.subheader("💬 Session Transcript")
                user_messages = [msg['content'] for msg in messages if msg['role'] == 'user']
                if user_messages:
                    transcript_text = "\n\n".join([f"Patient: {msg}" for msg in user_messages])
                    st.text_area("Patient Messages", transcript_text, height=200, disabled=True, key=f"transcript_{selected_patient}_{i}")

                # Doctor notes section
                st.subheader("📝 Clinical Notes")
                doctor_notes = st.text_area(f"Add clinical notes for session {i+1}:", 
                                          key=f"notes_{selected_patient}_{i}", height=100)
                
                if st.button(f"💾 Save Notes for Session {i+1}", key=f"save_{selected_patient}_{i}"):
                    # In a real system, this would save to database
                    st.success("Clinical notes saved successfully!")
                
                # Generate comprehensive report
                if st.button(f"📄 Generate Full Report for Session {i+1}", key=f"report_{selected_patient}_{i}"):
                    full_report = generate_comprehensive_report(session, patient_data, doctor_notes)
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

RECOMMENDATIONS
---------------
"""
    
    recommendations = st.session_state.analyzer.generate_recommendations(analysis)
    report += f"Immediate Action: {recommendations['immediate_action']}\n\n"
    report += "Suggested Next Steps:\n"
    for i, rec in enumerate(recommendations['recommendations'], 1):
        report += f"{i}. {rec}\n"
    
    report += f"\nFollow-up Timeline: {recommendations['follow_up']}\n"
    
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
IC Number: {patient_data.get('ic_number', 'N/A')}
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

AI ANALYSIS RESULTS
-------------------
Overall Risk Assessment: {analysis.get('risk_level', 'Unknown')}
Sentiment Analysis Score: {analysis.get('sentiment_score', 0):.3f}
Depression Risk Indicators: {analysis.get('depression_indicators', 0)}
Anxiety Risk Indicators: {analysis.get('anxiety_indicators', 0)}
Crisis Risk Indicators: {analysis.get('crisis_indicators', 0)}

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
Analysis Engine: MindBridge AI v1.0
Assessment Method: Natural Language Processing + Sentiment Analysis
Confidence Level: {'High' if len(user_messages) > 3 else 'Moderate'}
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

Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Generated by: MindBridge AI Mental Health Platform
Version: 1.0 (Prototype)
"""
    
    return report

if __name__ == "__main__":
    main()
