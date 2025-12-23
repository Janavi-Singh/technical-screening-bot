import json
import os
import re
import google.genai as genai
from typing import List, Dict, Any
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API Key not found! Please check your .env file.")

client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-2.0-flash"

WEIGHTS = {
    "syntax": 15,
    "logic": 40,
    "efficiency": 20,
    "quality": 15,
    "style": 10,
}

class AIEvaluator:
    
    def quantitative_check(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        results = payload.get("test_case_results", [])
        total = len(results)
        passed = sum(1 for r in results if r.get("pass", False))

        if payload.get("syntax_error"):
            return {"status": "Syntax Error", "pass_rate": 0.0, "passed_tests": 0, "total_tests": total}

        pass_rate = (passed / total) * 100 if total > 0 else 0.0
        status = "Runtime Error" if payload.get("runtime_error") else ("Partial Pass" if pass_rate < 100 else "Completed")

        return {
            "status": status, 
            "pass_rate": round(pass_rate, 2),
            "passed_tests": passed,
            "total_tests": total
        }

    def qualitative_check(self, payload: Dict[str, Any], quant: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("syntax_error"):
            return self._get_fallback_error("Syntax Error - Code did not compile.")

        prompt = f"""
        You are a Senior Technical Interviewer. Analyze this code submission deeply.

        ### CONTEXT
        Problem: {payload.get('ai_generated_coding_question')}
        Language: {payload.get('language')}
        Pass Rate: {quant['pass_rate']}% ({quant['passed_tests']}/{quant['total_tests']} tests passed)

        ### CANDIDATE CODE
        ```
        {payload.get('User_code')}
        ```

        ### TASK
        Provide a structured technical review.
        1. **Complexity:** Identify Time & Space complexity (Big O).
        2. **Code Quality:** Critique naming, modularity, and language idioms.
        3. **Logic:** If tests failed, explain WHY. If they passed, suggest an optimization.

        ### OUTPUT FORMAT (Strict JSON)
        {{
            "technical_analysis": {{
                "efficiency_score": (int 1-10),
                "style_score": (int 1-10),
                "time_complexity": "string",
                "space_complexity": "string",
                "critique": "1-2 sentences on what is good/bad technically."
            }},
            "feedback_for_candidate": {{
                "what_went_well": "string",
                "what_to_improve": "string"
            }}
        }}
        """
        return self._call_ai(prompt)

    def synthesize_final_report(self, individual_reports: List[Dict[str, Any]], overall_score: float) -> Dict[str, Any]:
        summary_context = ""
        for i, rep in enumerate(individual_reports):
            tech = rep['ai_analysis'].get('technical_analysis', {})
            summary_context += f"""
            Q{i+1}: {rep['question']} ({rep['language']})
            - Score: {rep['final_score']}
            - Pass Rate: {rep['metrics']['pass_rate']}%
            - Efficiency: {tech.get('efficiency_score', 'N/A')}
            - Critique: {tech.get('critique', 'N/A')}
            """

        prompt = f"""
        You are the Hiring Manager. Write a final decision based on these metrics.

        ### CANDIDATE DATA
        Overall Score: {overall_score}/100
        {summary_context}

        ### TASK
        Return strict JSON (no markdown).

        {{
            "recruiter_executive_summary": {{
                "hiring_decision": "Strong Hire/Hire/Weak Hire/No Hire",
                "candidate_level_assessment": "Junior/Mid-Level/Senior",
                "final_conclusion": "2 sentences justifying the decision."
            }},
            "candidate_holistic_feedback": {{
                "major_strength": "The one thing they did best.",
                "major_weakness": "The one thing they must fix.",
                "final_recommendation": "One specific topic to study."
            }}
        }}
        """
        return self._call_ai(prompt)

    def _call_ai(self, prompt: str) -> Dict[str, Any]:
        try:
            # FIX: Using a simple dictionary for config avoids the 'AttributeError'
            # associated with the GenerationConfig object in some SDK versions.
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    'response_mime_type': 'application/json'
                }
            )
            
            raw = response.text.strip()
            
            # Robust Regex to find JSON content inside code blocks or plain text
            match = re.search(r"```(?:json)?\s*(.*)\s*```", raw, re.DOTALL)
            if match: 
                raw = match.group(1)
            
            return json.loads(raw)

        except Exception as e:
            print(f"AI GENERATION ERROR: {e}")
            return {}

    def _get_fallback_error(self, message):
        return {
            "technical_analysis": {"efficiency_score": 5, "style_score": 5, "critique": "AI Error"},
            "feedback_for_candidate": {"what_went_well": "N/A", "what_to_improve": f"Error: {message}"},
            "recruiter_executive_summary": {"hiring_decision": "Undetermined", "final_conclusion": "AI Service Unavailable"},
            "candidate_holistic_feedback": {"major_strength": "N/A", "major_weakness": "N/A"}
        }

    def score(self, quant: Dict[str, Any], qual: Dict[str, Any], syntax_error: bool) -> float:
        if syntax_error: return 0.0
        
        eff_score = qual.get("technical_analysis", {}).get("efficiency_score", 0)
        style_score = qual.get("technical_analysis", {}).get("style_score", 0)

        total = WEIGHTS["syntax"]
        total += (quant["pass_rate"] / 100) * WEIGHTS["logic"]
        total += (eff_score / 10) * WEIGHTS["efficiency"]
        total += (style_score / 10) * (WEIGHTS["quality"] + WEIGHTS["style"])
        return round(total, 2)


class ReportGenerator:
    def __init__(self):
        self.evaluator = AIEvaluator()

    def generate(self, payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
        reports = []
        scores = []

        # 1. Analyze Individual Questions
        for payload in payloads:
            quant = self.evaluator.quantitative_check(payload)
            qual = self.evaluator.qualitative_check(payload, quant)
            final_score = self.evaluator.score(quant, qual, payload.get("syntax_error", False))

            scores.append(final_score)
            reports.append({
                "question": payload["ai_generated_coding_question"],
                "language": payload["language"],
                "final_score": final_score,
                "metrics": quant,
                "ai_analysis": qual,
            })

        # 2. Calculate Overall Average
        overall_avg = round(sum(scores) / len(scores), 2) if scores else 0.0

        # 3. Generate Executive Summary
        final_summary = self.evaluator.synthesize_final_report(reports, overall_avg)

        # 4. Final Structure
        return {
            "detailed_question_analysis": reports,
            "final_conclusive_report": {
                "overall_score": overall_avg,
                "status": "Passed" if overall_avg >= 70 else "Failed",
                "recruiter_summary": final_summary.get("recruiter_executive_summary", {}),
                "candidate_growth_plan": final_summary.get("candidate_holistic_feedback", {})
            }
        }

if __name__ == "__main__":
    MULTI_QUESTION_PAYLOAD = [
        # PYTHON 
    {
        "ai_generated_coding_question": "Q1: Implement a function to reverse a string.",
        "language": "Python",
        "User_code": "def reverse_string(s): return s[::-1]",
        "test_case_results": [
            {"input": "hello", "expected": "olleh", "pass": True},
            {"input": "world", "expected": "dlrow", "pass": True}
        ],
        "syntax_error": False,
        "runtime_error": False
    },

    # JAVA 
    {
        "ai_generated_coding_question": "Q2: Write a function to find the factorial of a number.",
        "language": "Java",
        "User_code": (
            "class Solution { "
            "static int factorial(int n) { "
            "if (n < 0) return -1; "
            "if (n == 0) return 1; "
            "return n * factorial(n - 1); "
            "} }"
        ),
        "test_case_results": [
            {"input": 5, "expected": 120, "pass": True},
            {"input": -1, "expected": -1, "pass": True},
            {"input": 0, "expected": 1, "pass": True}
        ],
        "syntax_error": False,
        "runtime_error": False
    },

    # C++ Question
    {
        "ai_generated_coding_question": "Q3: Implement a function to reverse a string.",
        "language": "C++",
        "User_code": (
            "#include <string>\n"
            "using namespace std;\n"
            "string reverseString(string s) {\n"
            "    reverse(s.begin(), s.end());\n"
            "    return s;\n"
            "}"
        ),
        "test_case_results": [
            {"input": "hello", "expected": "olleh", "pass": True},
            {"input": "world", "expected": "dlrow", "pass": True}
        ],
        "syntax_error": False,
        "runtime_error": False
    }
    ]

    generator = ReportGenerator()
    final_report = generator.generate(MULTI_QUESTION_PAYLOAD)
    print(json.dumps(final_report, indent=4))