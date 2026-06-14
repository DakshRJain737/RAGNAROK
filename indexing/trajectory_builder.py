import config
from pipeline.schemas import CandidateFeatureVector


class TrajectoryAnalyzer:

    # --------------------------------------------------
    # TENURE (Pure Utility)
    # --------------------------------------------------
    @staticmethod
    def calculate_tenure_metrics(candidate: CandidateFeatureVector) -> tuple[float, float, float]:
        """
        Calculates all tenure-related stats in a single pass to optimize CPU execution.
        Returns: (avg_tenure, stability_score, job_hopper_flag)
        """
        history = candidate.career_history
        if not history:
            return 0.0, 0.0, 1.0  # No history means 0 tenure, 0 stability, is a job hopper

        total_months = sum(job.duration_months for job in history)
        avg_tenure = (total_months / len(history)) / 12.0
        
        stability_score = min(avg_tenure / 3.0, 1.0)
        is_job_hopper = float(avg_tenure < 1.5)
        
        return avg_tenure, stability_score, is_job_hopper

    # --------------------------------------------------
    # CONSULTING & PRODUCT EXPERIENCE (Combined Pass)
    # --------------------------------------------------
    @staticmethod
    def analyze_career_history(candidate: CandidateFeatureVector) -> tuple[float, float]:
        """
        Scans career history ONCE to determine both consulting and product tracks.
        Returns: (consulting_only_flag, has_product_exp_flag)
        """
        history = candidate.career_history
        if not history:
            return 0.0, 0.0

        has_companies = False
        all_companies_are_consulting = True
        has_product_experience = 0.0

        # Pre-cache lowered configuration sets for instant O(1) lookups
        consulting_firms = getattr(config, "_cached_consulting", None)
        if consulting_firms is None:
            config._cached_consulting = consulting_firms = {f.lower().strip() for f in config.CONSULTING_FIRMS}
            
        product_industries = getattr(config, "_cached_product", None)
        if product_industries is None:
            config._cached_product = product_industries = {i.lower().strip() for i in config.PRODUCT_INDUSTRIES}

        for job in history:
            # 1. Check Consulting status
            if job.company:
                has_companies = True
                company_clean = job.company.lower().strip()
                if company_clean not in consulting_firms:
                    all_companies_are_consulting = False

            # 2. Check Product Experience status
            if job.industry:
                industry_clean = job.industry.lower().strip()
                if industry_clean in product_industries:
                    has_product_experience = 1.0

        is_consulting_only = float(has_companies and all_companies_are_consulting)
        return is_consulting_only, has_product_experience

    # --------------------------------------------------
    # YOE SCORE
    # --------------------------------------------------
    @staticmethod
    def yoe_score(candidate: CandidateFeatureVector) -> float:
        yoe = candidate.years_of_experience

        if config.YOE_BAND_IDEAL_MIN <= yoe <= config.YOE_BAND_IDEAL_MAX:
            return 1.0

        if yoe < config.YOE_BAND_IDEAL_MIN:
            return max(0.0, (yoe / config.YOE_BAND_IDEAL_MIN) ** 2)

        if yoe <= config.YOE_BAND_MAX:
            excess = yoe - config.YOE_BAND_IDEAL_MAX
            width = config.YOE_BAND_MAX - config.YOE_BAND_IDEAL_MAX
            return max(0.0, 1.0 - (excess / width))

        return 0.25

    # --------------------------------------------------
    # CAREER SCORE
    # --------------------------------------------------
    def career_score(self, candidate: CandidateFeatureVector) -> float:
        traj = self.build_feature_vector(candidate)

        score = (
            0.40 * traj["yoe_score"]
            + 0.30 * traj["product_experience"]
            + 0.30 * traj["stability_score"]
        )

        if traj["consulting_only"] == 1.0:
            score *= config.CONSULTING_ONLY_PENALTY

        return round(score, 4)

    # --------------------------------─
    # FEATURE VECTOR
    # --------------------------------─
    def build_feature_vector(self, candidate: CandidateFeatureVector) -> dict:
        # Calculate structural metrics in combined efficient loops
        avg_tenure, stability, job_hopper = self.calculate_tenure_metrics(candidate)
        is_consulting, product_exp = self.analyze_career_history(candidate)

        return {
            "yoe_score": self.yoe_score(candidate),
            "avg_tenure": avg_tenure,
            "stability_score": stability,
            "job_hopper": job_hopper,
            "consulting_only": is_consulting,
            "product_experience": product_exp,
        }
    
    def build_all_feature_vector(self, candidates: list[CandidateFeatureVector]) -> list[dict]:
        return [self.build_feature_vector(c) for c in candidates]
