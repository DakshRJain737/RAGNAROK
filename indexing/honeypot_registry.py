from pipeline.schemas import CandidateFeatureVector
import config

class HoneypotFilter:
    
    # 2. Check if skills are actually mentioned in career descriptions
    def __filter2(self, candidate: CandidateFeatureVector) -> bool:
        skills = candidate.skills
        if not skills or not candidate.career_history:
            return False

        # This prevents looping through career history repeatedly for every single skill.
        full_career_text = " ".join(
            career.description.lower() 
            for career in candidate.career_history 
            if career.description
        )

        if not full_career_text:
            return False

        # Check for matching skills
        for skill in skills:
            if skill.proficiency and skill.proficiency.lower() in {"expert", "advanced", "intermediate"}:
                skill_name = skill.name.lower() if skill.name else ""
                if skill_name and skill_name in full_career_text:
                    return True
        return False

    # 3. Profile completeness score < threshold with suspiciously many skills
    def __filter3(self, candidate: CandidateFeatureVector) -> bool:
        profile_incomplete = candidate.signals.profile_completeness_score < config.HONEYPOT_COMPLETENESS_THRESHOLD
        too_many_skills = len(candidate.skills) > config.HONEYPOT_SKILLS_STUFFING_COUNT
        return profile_incomplete or too_many_skills

    # 4. Salary min > max anomaly
    def __filter4(self, candidate: CandidateFeatureVector) -> bool:
        return candidate.signals.expected_salary_min_lpa > candidate.signals.expected_salary_max_lpa

    # 5. Experience discrepancy check
    def __filter5(self, candidate: CandidateFeatureVector) -> bool:
        total_duration_months = sum(
            career.duration_months 
            for career in candidate.career_history 
            if career.duration_months
        )
        total_duration_years = total_duration_months / 12.0

        return (total_duration_years - config.HONEYPOT_YOE_DISCREPANCY_YEARS) > candidate.years_of_experience

    def run_honeypot_filters(self, candidates: list[CandidateFeatureVector]) -> None:
        for candidate in candidates:
            if self.__filter4(candidate):
                candidate.is_honeypot = True
                continue
            if self.__filter5(candidate):
                candidate.is_honeypot = True
                continue
            if self.__filter3(candidate):
                candidate.is_honeypot = True
                continue
            if self.__filter2(candidate):
                candidate.is_honeypot = True
                continue
