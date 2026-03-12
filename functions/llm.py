import json
import os
from dotenv import load_dotenv
from groq import Groq

# Load .env
load_dotenv()

# Initialize Groq client (reads GROQ_API_KEY from environment)
client = Groq()

print("Groq client initialized.")

def llm_call(resume_text, job, system_prompt, user_prompt, expect_json=False):
    """
    Uses Groq openai/gpt-oss-120b model
    to score and analyze a job match.
    """

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_completion_tokens=2048,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
    )

    # full_response = completion.choices[0].message.content
    full_response = ""
    print("Full response:", full_response)

    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            full_response += content

   
    # Try parsing JSON safely
    if expect_json:
            try:
                return json.loads(full_response)
            except:
                print("⚠ JSON parsing failed. Returning raw output.")
                return {"raw_output": full_response}

    return full_response


    
