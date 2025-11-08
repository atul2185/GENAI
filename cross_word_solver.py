from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

# --- Fix 1: Load environment variables and check for API key ---
# Load environment variables from a .env file (e.g., GROQ_API_KEY="...")
load_dotenv() 

# Check if the API key is available
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file.")

# Initialize the ChatGroq model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

def solve_crossword_clue(clue, num_letters):
    """
    Uses the Groq model to solve a single crossword clue.
    
    Args:
        clue (str): The crossword clue text.
        num_letters (int): The required length of the answer word.
        
    Returns:
        tuple[str, str] or tuple[None, None]: The solved answer and the finish reason, 
                                               or None, None on failure.
    """
    # --- Fix 2: Enhanced system prompt for better single-word output ---
    # Instruct the model to strictly output only the single word.
    messages=[
        {"role": "system", "content": "You are a helpful assistant that solves crossword clues. You must output ONLY the single word answer and nothing else."},
        {"role": "user", "content": f"Solve this crossword clue: The clue is '{clue}' and it is a {num_letters} letters word. The answer is :"}
    ]
    
    try:
        # --- Fix 3: Adjust max_tokens to accommodate the longest expected answer (7 letters) ---
        # The longest word is 'jupiter' (7 letters). Setting max_tokens to 10 ensures 
        # the full word is returned without being cut off.
        # Original: max_tokens=3 (This caused truncation for 'apple', 'paris', and 'jupiter')
        response = model.invoke(messages, max_tokens=10) 
        
        # Clean up the output: strip whitespace and convert to lowercase for comparison
        answer = response.content.strip().lower()
        finish_reason = response.response_metadata.get('finish_reason', 'N/A').strip().lower()
        
        return answer, finish_reason
        
    except Exception as e:
        print(f"Error solving clue '{clue}': {e}")
        return None, None
            

def main():
    """
    Main function to run the batch test of crossword clues.
    """
    # Define the test cases
    clues=[
        {"clue":"A fruit that keeps the doctor away","num_letters":5,"expected":"apple"},
        {"clue":"The largest planet in our solar system","num_letters":7,"expected":"jupiter"},
        {"clue":"The capital city of France","num_letters":5,"expected":"paris"}
    ]
    
    correct=0
    print("--- Groq Crossword Solver Test Results ---")
    
    # Iterate through the clues, solve them, and check the results
    for clue in clues:
        answer, finish_reason = solve_crossword_clue(clue["clue"], clue["num_letters"])
        
        if answer is not None:
            # Check if the extracted answer matches the expected answer
            is_correct = "✅ Correct" if answer == clue["expected"] else "❌ Incorrect"
            
            print(f"\n{is_correct}:")
            print(f"  Clue: {clue['clue']}")
            print(f"  Expected: {clue['expected']}")
            print(f"  Got: {answer} (Finish Reason: {finish_reason})")
            
            if answer == clue["expected"]:
                correct+=1
        else:
            print(f"\nError processing clue: {clue['clue']}")

    print("\n--- Summary ---")
    print(f"Solved **{correct}** out of **{len(clues)}** clues correctly.")
    print("-----------------")

if __name__=="__main__":
    main()