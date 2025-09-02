from utils.questions import questions
from pipeline import answer_question

for q in questions[:1]:
    result = answer_question(q["question"])
    print({"question_id": q["question_id"], "question": q["question"], **result})
