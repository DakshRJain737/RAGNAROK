from pipeline.schemas import CandidateFeatureVector
import config

class HoneypotFilter:

    # 1. Exp at company > founding delta
    # Rule 1: Experience at company exceeds company's plausible founding delta.
    # If career_history shows N months at a company but start_date predates
    # any reasonable founding, flag. We use a conservative 6-month buffer.
    def filter1(candidates : CandidateFeatureVector) -> bool:
        pass
    
    # 2. expert proficiency + 0 months duration
    @staticmethod
    def __filter2(candidate : CandidateFeatureVector) -> bool:
        skills = candidate.skills
        for skill in skills:
            if skill.proficiency.lower() in ["expert", "advanced", "intermediate"]:
                skill_name = skill.name.lower()
                has_exact_match = any(
                    skill_name in career.description.lower() 
                    for career in candidate.career_history
                    )
                
                if has_exact_match:
                    return True
        return False

    # 3. Profile completeness score < threshold with suspiciously many skills
    def __filter3(candidate : CandidateFeatureVector) -> bool:
        profile_incomplete = candidate.signals.profile_completeness_score < config.HONEYPOT_COMPLETENESS_THRESHOLD
        too_many_skills = len(candidate.skills) > config.HONEYPOT_SKILLS_STUFFING_COUNT > config.HONEYPOT_SKILLS_STUFFING_COUNT
        return profile_incomplete or too_many_skills

    # 4. salary min > max anomaly
    @staticmethod
    def __filter4(candidate : CandidateFeatureVector) -> bool:
        return candidate.signals.expected_salary_min_lpa > candidate.signals.expected_salary_max_lpa
            

    # 5. O(1) registry lookup · removes before cross-encoder
    def __filter5(candidate : CandidateFeatureVector) -> bool:
        total_duration = 0
        for career in candidate.career_history:
            total_duration += career.duration_months
        total_duration_years = total_duration/12

        return total_duration_years - config.HONEYPOT_YOE_DISCREPANCY_YEARS > candidate.years_of_experience
        
    @staticmethod
    def run_honeypot_filters(candidates : list[CandidateFeatureVector]):
        for candidate in candidates:
            if HoneypotFilter.__filter4(candidate):
                candidate.is_honeypot = True
                continue
            if HoneypotFilter.__filter3(candidate):
                candidate.is_honeypot = True
                continue
            if HoneypotFilter.__filter5(candidate):
                candidate.is_honeypot = True
                continue
            if HoneypotFilter.__filter2(candidate):
                candidate.is_honeypot = True
                continue
