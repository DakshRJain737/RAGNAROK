from __future__ import annotations

import logging

from pipeline.schemas import CandidateFeatureVector

logger = logging.getLogger(__name__)


class HoneypotCleanup:

    def cleanup_candidates(
        self,
        candidates: list[CandidateFeatureVector],
    ) -> list[CandidateFeatureVector]:
        honeypots = [c for c in candidates if c.is_honeypot]
        clean = [c for c in candidates if not c.is_honeypot]

        if honeypots:
            honeypot_ids = [c.candidate_id for c in honeypots]
            logger.info(
                "HoneypotCleanup: removed %d honeypot candidate(s) from pool "
                "(remaining=%d). IDs: %s",
                len(honeypots),
                len(clean),
                honeypot_ids,
            )
        else:
            logger.debug(
                "HoneypotCleanup: no honeypots detected in pool of %d candidates.",
                len(candidates),
            )

        return clean
