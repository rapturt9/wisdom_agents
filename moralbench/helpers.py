##############################################
# MoralBench_AgentEnsembles : 
# NOTE: there is a diffference here with the
# original code, where the repo is cloned
##############################################
# Clone the repository
repo_url = "https://github.com/MartinLeitgab/MoralBench_AgentEnsembles/"
repo_dir = "../MoralBench_AgentEnsembles"

# Check if directory already exists to avoid errors
if not os.path.exists(repo_dir):
    subprocess.run(["git", "clone", repo_url])
    print(f"Repository cloned to {repo_dir}")
else:
    print(f"Repository directory {repo_dir} already exists")

# Add the repository to Python path instead of changing directory
repo_path = os.path.abspath(repo_dir)
sys.path.append(repo_path)
print(f"Added {repo_path} to Python path")

########################################################
# Questions for MoralBench_AgentEnsembles : 
########################################################
def get_question_count(category_folder):
    """
    Get the number of questions in a specific category folder.

    Args:
        category_folder (str): The name of the category folder (e.g., '6_concepts', 'MFQ_30')

    Returns:
        int: Number of questions in the folder
    """
    questions_path = os.path.join('questions', category_folder)
    if not os.path.exists(questions_path):
        print(f"Category folder {category_folder} does not exist!")
        return 0

    question_files = [f for f in os.listdir(questions_path) if f.endswith('.txt')]
    return len(question_files)

def list_categories():
    """
    List all available question categories.

    Returns:
        list: A list of category folder names
    """
    if not os.path.exists('questions'):
        print("Questions directory not found!")
        return []

    categories = [d for d in os.listdir('questions') if os.path.isdir(os.path.join('questions', d))]
    return categories

def load_question_answer(category_folder, index):
    """
    Load a question and its possible answers using an index.

    Args:
        category_folder (str): The name of the category folder (e.g., '6_concepts', 'MFQ_30')
        index (int): The index of the question (0-based)

    Returns:
        dict: A dictionary containing question text and possible answers with scores
    """
    questions_path = os.path.join('questions', category_folder)
    if not os.path.exists(questions_path):
        print(f"Category folder {category_folder} does not exist!")
        return None

    # Get all question files and sort them
    question_files = sorted([f for f in os.listdir(questions_path) if f.endswith('.txt')])

    if index < 0 or index >= len(question_files):
        print(f"Index {index} is out of range! Valid range: 0-{len(question_files)-1}")
        return None

    # Get question filename and ID
    question_file = question_files[index]
    question_id = os.path.splitext(question_file)[0]

    # Read question content
    question_path = os.path.join(questions_path, question_file)
    with open(question_path, 'r') as f:
        question_text = f.read()

    # Load answers from JSON
    answers_path = os.path.join('answers', f"{category_folder}.json")
    if not os.path.exists(answers_path):
        print(f"Answers file for {category_folder} does not exist!")
        return {'question_id': question_id, 'question_text': question_text, 'answers': None}

    with open(answers_path, 'r') as f:
        all_answers = json.load(f)

    # Get answers for this question
    question_answers = all_answers.get(question_id, {})

    return {
        'question_id': question_id,
        'question_text': question_text,
        'answers': question_answers
    }

def display_question_info(question_data):
    """
    Display formatted information about a question.

    Args:
        question_data (dict): Question data from load_question_answer function
    """
    if not question_data:
        return

    print(f"\n=== Question ID: {question_data['question_id']} ===")
    print(f"\n{question_data['question_text']}")

    if question_data['answers']:
        print("\nPossible answers and their scores:")
        for option, score in question_data['answers'].items():
            print(f"Option {option}: {score} points")
    else:
        print("\nNo scoring information available for this question.")

def get_question(number):
  # enumerate across categories and questions
  categories = list_categories()
  num_questions = 0
  for category in categories:
    for i in range(get_question_count(category)):
      num_questions += 1
      if num_questions == number:
        return load_question_answer(category, i)
  return None

def get_total_question_count():
  categories = list_categories()
  total = 0
  for category in categories:
    total += get_question_count(category)
  return total

# List all available categories
categories = list_categories()
print("Available question categories:")
for i, category in enumerate(categories):
    count = get_question_count(category)
    print(f"{i+1}. {category} ({count} questions)")

# Example usage - load the first question from the first category
if categories:
    first_category = categories[0]
    first_question = load_question_answer(first_category, 0)
    display_question_info(first_question)

    # Example of how to access question fields directly
    print("\nAccessing question fields directly:")
    print(f"Question ID: {first_question['question_id']}")
    print(f"Question text length: {len(first_question['question_text'])} characters")
    print(f"Answer options: {list(first_question['answers'].keys())}")