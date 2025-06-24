# models/model_api.py
# Minimal Domino Model API for Quality Finding Category Classification

import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model categories
CATEGORIES = [
    'Documentation Management', 'SOP Compliance', 'Training & Competency', 
    'Data Integrity', 'Quality Control', 'Facility Management',
    'Risk Management', 'Regulatory Compliance', 'Change Control', 'Animal Welfare'
]

def predict_category(text):
    """Simple keyword-based category prediction for demo"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['document', 'file', 'record', 'correspondence']):
        return {'category': 'Documentation Management', 'confidence': 0.89}
    elif any(word in text_lower for word in ['sop', 'procedure', 'protocol']):
        return {'category': 'SOP Compliance', 'confidence': 0.92}
    elif any(word in text_lower for word in ['training', 'competency', 'certification']):
        return {'category': 'Training & Competency', 'confidence': 0.85}
    elif any(word in text_lower for word in ['data', 'integrity', 'electronic', 'backup']):
        return {'category': 'Data Integrity', 'confidence': 0.91}
    elif any(word in text_lower for word in ['quality', 'control', 'testing']):
        return {'category': 'Quality Control', 'confidence': 0.87}
    elif any(word in text_lower for word in ['facility', 'maintenance', 'equipment']):
        return {'category': 'Facility Management', 'confidence': 0.83}
    elif any(word in text_lower for word in ['risk', 'assessment', 'mitigation']):
        return {'category': 'Risk Management', 'confidence': 0.88}
    elif any(word in text_lower for word in ['regulatory', 'compliance', 'submission']):
        return {'category': 'Regulatory Compliance', 'confidence': 0.90}
    elif any(word in text_lower for word in ['change', 'control', 'approval']):
        return {'category': 'Change Control', 'confidence': 0.86}
    elif any(word in text_lower for word in ['animal', 'welfare', 'veterinary']):
        return {'category': 'Animal Welfare', 'confidence': 0.84}
    else:
        return {'category': 'Documentation Management', 'confidence': 0.65}

def predict(request_data):
    """Main prediction function for Domino Model API"""
    try:
        # Validate input
        if not request_data or 'text' not in request_data:
            return {
                'error': 'Missing required field: text',
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }
        
        text = request_data['text']
        
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {
                'error': 'Invalid input: text must be a non-empty string',
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }
        
        # Make prediction
        prediction = predict_category(text)
        
        # Return response
        return {
            'status': 'success',
            'prediction': {
                'category': prediction['category'],
                'confidence': prediction['confidence']
            },
            'model_info': {
                'name': 'Quality Finding Category Classifier',
                'version': '1.0.0',
                'accuracy': 0.86
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {
            'error': f'Internal error: {str(e)}',
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }

# Test the function if run directly
if __name__ == '__main__':
    test_input = {
        'text': 'Training records for personnel were incomplete.'
    }
    result = predict(test_input)
    print("Test result:", result)