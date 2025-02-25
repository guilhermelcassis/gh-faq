import requests
import json
import time

def test_health():
    response = requests.get('http://localhost:8000/health')  # Adjust the endpoint as necessary
    print(f"Response Status Code: {response.status_code}")  # Check the status code
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: Received status code {response.status_code}")
        print("Response Text:", response.text)  # Print the response text for debugging
    
    return response.status_code == 200

def test_questions(questions):
    """Test a list of questions."""
    for question in questions:
        print(f"Question: {question}")
        
        start_time = time.time()
        response = requests.post(
            "http://localhost:8000/ask",
            json={"question": question}
        )
        end_time = time.time()
        
        print(f"Response Time: {end_time - start_time:.2f} seconds")
        print("Answer:")
        print(json.dumps(response.json(), indent=2))
        print("\n" + "-"*50 + "\n")
        
        # Small delay to avoid rate limiting
        time.sleep(1)

if __name__ == "__main__":
    # Test health endpoint
#    if not test_health():
#        print("Health check failed. Exiting.")
#        exit(1)
    
    # Test some questions
    test_questions([
        "What is Greenhouse?",
        "How much does it cost?",
        "Where is the school located?",
        "What will I learn at Greenhouse?",
        "How do I apply to a mission trip?",
        "What is the minimum age requirement?",
        "What should I pack for the school?",
        "Will there be mission trips?",
        "What will happen at the school?",
        "How do I get to the school?",
        "Where will GH be?",
        "Is it expensive?",
        "What will I do at school?"
    ]) 