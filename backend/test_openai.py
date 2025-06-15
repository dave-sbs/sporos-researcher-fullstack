import os
from openai import OpenAI
from dotenv import load_dotenv

def test_openai_connection():
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    try:
        # Test a simple completion
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "how's your day'"}
            ],
            max_tokens=10
        )
        
        print("\nAPI Test Results:")
        print("-----------------")
        print("✅ API Connection: Successful")
        print(f"Response: {response.choices[0].message.content}")
        
    except Exception as e:
        print("\n❌ Error testing OpenAI API:")
        print(f"Error message: {str(e)}")

if __name__ == "__main__":
    test_openai_connection() 