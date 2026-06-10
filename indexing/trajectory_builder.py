# trajectory_analyzer.py

import config
from pipeline.schemas import CandidateFeatureVector


class TrajectoryAnalyzer:

    # --------------------------------------------------
    # TENURE
    # --------------------------------------------------

    @staticmethod
    def average_tenure_years(candidate):

        history = candidate.career_history

        if not history:
            return 0.0

        total_months = sum(
            job.duration_months
            for job in history
        )

        return (total_months / len(history)) / 12.0

    # --------------------------------------------------
    # STABILITY SCORE
    # --------------------------------------------------

    @staticmethod
    def stability_score(candidate):

        tenure = (
            TrajectoryAnalyzer.average_tenure_years(candidate)
        )

        return min(
            tenure / 3.0,
            1.0
        )

    # --------------------------------------------------
    # JOB HOPPER
    # --------------------------------------------------

    @staticmethod
    def is_job_hopper(candidate):

        return (
            TrajectoryAnalyzer.average_tenure_years(candidate)
            < 1.5
        )

    # --------------------------------------------------
    # CONSULTING ONLY
    # --------------------------------------------------

    @staticmethod
    def is_consulting_only(candidate):

        companies = []

        for job in candidate.career_history:

            if not job.company:
                continue

            companies.append(
                job.company.lower().strip()
            )

        if not companies:
            return False

        return all(
            company in config.CONSULTING_FIRMS
            for company in companies
        )

    # --------------------------------------------------
    # PRODUCT EXPERIENCE
    # --------------------------------------------------

    @staticmethod
    def has_product_experience(candidate):

        for job in candidate.career_history:

            industry = (
                job.industry.lower().strip()
                if job.industry
                else ""
            )

            if industry in config.PRODUCT_INDUSTRIES:
                return True

        return False

    # --------------------------------------------------
    # YOE SCORE
    # --------------------------------------------------

    @staticmethod
    def yoe_score(candidate):

        yoe = candidate.years_of_experience

        if (
            config.YOE_BAND_IDEAL_MIN
            <= yoe
            <= config.YOE_BAND_IDEAL_MAX
        ):
            return 1.0

        if yoe < config.YOE_BAND_IDEAL_MIN:

            return max(
                0.0,
                (yoe / config.YOE_BAND_IDEAL_MIN)**2
            )

        if yoe <= config.YOE_BAND_MAX:

            excess = (
                yoe
                - config.YOE_BAND_IDEAL_MAX
            )

            width = (
                config.YOE_BAND_MAX
                - config.YOE_BAND_IDEAL_MAX
            )

            return max(
                0.0,
                1.0 - (excess / width)
            )

        return 0.25

    # --------------------------------------------------
    # CAREER SCORE
    # --------------------------------------------------

    @staticmethod
    def career_score(candidate):

        traj = (
            TrajectoryAnalyzer.build_feature_vector(
                candidate
            )
        )

        score = (
            0.40 * traj["yoe_score"]
            + 0.30 * traj["product_experience"]
            + 0.30 * traj["stability_score"]
        )

        if traj["consulting_only"] == 1.0:
            score *= config.CONSULTING_ONLY_PENALTY

        return round(score, 4)

    # --------------------------------------------------
    # FEATURE VECTOR
    # --------------------------------------------------

    @staticmethod
    def build_feature_vector(
        candidate: CandidateFeatureVector
    ):

        return {

            "yoe_score":
                TrajectoryAnalyzer.yoe_score(candidate),

            "avg_tenure":
                TrajectoryAnalyzer.average_tenure_years(
                    candidate
                ),

            "stability_score":
                TrajectoryAnalyzer.stability_score(
                    candidate
                ),

            "job_hopper":
                float(
                    TrajectoryAnalyzer.is_job_hopper(
                        candidate
                    )
                ),

            "consulting_only":
                float(
                    TrajectoryAnalyzer.is_consulting_only(
                        candidate
                    )
                ),

            "product_experience":
                float(
                    TrajectoryAnalyzer.has_product_experience(
                        candidate
                    )
                ),
        }