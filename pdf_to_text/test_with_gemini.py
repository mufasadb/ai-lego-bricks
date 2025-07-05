#!/usr/bin/env python3
"""
Test PDF-to-Text with Gemini Vision manually
"""

import os
import sys
import base64
from dotenv import load_dotenv
import pymupdf
import httpx

# Load environment variables
load_dotenv()

def test_gemini_vision(pdf_path):
    """Test Gemini Vision on PDF pages"""
    
    api_key = os.getenv('GOOGLE_AI_STUDIO_KEY')
    if not api_key:
        print("❌ GOOGLE_AI_STUDIO_KEY not found in environment")
        return
    
    print(f"🔍 Testing Gemini Vision on: {pdf_path}")
    
    try:
        # Open PDF and convert first page to image
        with pymupdf.open(pdf_path) as doc:
            print(f"📄 PDF has {doc.page_count} pages")
            
            # Convert first page to image
            page = doc[0]
            pix = page.get_pixmap(dpi=150)
            img_data = pix.tobytes("png")
            img_base64 = base64.b64encode(img_data).decode()
            
            print(f"🖼️  Converted first page to image ({len(img_base64)} characters base64)")
            
            # Call Gemini Vision API
            prompt = "Extract all text from this PDF page image. Maintain the original structure and formatting as much as possible."
            
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": img_base64
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 2048
                }
            }
            
            print("🤖 Calling Gemini Vision API...")
            
            with httpx.Client(timeout=30) as client:
                response = client.post(
                    "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
                    json=payload,
                    headers={"x-goog-api-key": api_key}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'candidates' in result and result['candidates']:
                        text = result['candidates'][0]['content']['parts'][0]['text']
                        print("✅ Gemini Vision extraction successful!")
                        print(f"📝 Extracted text ({len(text)} characters):")
                        print("=" * 50)
                        print(text)
                        print("=" * 50)
                        return text
                    else:
                        print(f"❌ No candidates in response: {result}")
                else:
                    print(f"❌ API error {response.status_code}: {response.text}")
                    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_gemini_text_enhancement(text):
    """Test Gemini text enhancement"""
    
    api_key = os.getenv('GOOGLE_AI_STUDIO_KEY')
    if not api_key:
        print("❌ GOOGLE_AI_STUDIO_KEY not found in environment")
        return
    
    print(f"🔍 Testing Gemini text enhancement...")
    
    try:
        prompt = f"""Please improve the formatting and structure of this extracted PDF text. 
Fix any OCR errors, improve paragraph breaks, restore proper formatting, and make it more readable.
Keep the content exactly the same but improve the presentation:

{text[:4000]}"""  # Limit text length for API
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 2048
            }
        }
        
        print("🤖 Calling Gemini Text API...")
        
        with httpx.Client(timeout=30) as client:
            response = client.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
                json=payload,
                headers={"x-goog-api-key": api_key}
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    enhanced_text = result['candidates'][0]['content']['parts'][0]['text']
                    print("✅ Gemini text enhancement successful!")
                    print(f"📝 Enhanced text ({len(enhanced_text)} characters):")
                    print("=" * 50)
                    print(enhanced_text)
                    print("=" * 50)
                    return enhanced_text
                else:
                    print(f"❌ No candidates in response: {result}")
            else:
                print(f"❌ API error {response.status_code}: {response.text}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test both PDFs
    pdfs = [
        "../_Infrastructure-040725-005140.pdf",
        "../zoom invoice august.pdf"
    ]
    
    for pdf_path in pdfs:
        if os.path.exists(pdf_path):
            print(f"\n{'='*60}")
            print(f"Testing: {pdf_path}")
            print('='*60)
            
            # Test vision extraction
            vision_text = test_gemini_vision(pdf_path)
            
            if vision_text:
                # Test text enhancement
                test_gemini_text_enhancement(vision_text)
        else:
            print(f"❌ File not found: {pdf_path}")