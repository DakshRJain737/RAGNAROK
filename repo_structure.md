# Repository Structure

Here is the visual directory tree for your repository (ignoring `.git`, `.venv`, and `__pycache__`):

```text
RAGnarok/
├── .gitignore
├── Daily_Task(sprint planner).html
├── LICENSE
├── Optimised_Pipeline.html
├── README.md
├── api
│   ├── main.py
│   ├── middleware.py
│   ├── routes
│   │   ├── __init__.py
│   │   ├── health.py
│   │   └── rank.py
│   └── schemas.py
├── build_indexes.py
├── config.py
├── data
│   ├── .gitkeep
│   ├── candidates.jsonl
│   ├── indexes
│   │   ├── bm25.pkl
│   │   ├── candidate_ids.npy
│   │   ├── faiss.index
│   │   ├── feature_ids.npy
│   │   ├── features.npy
│   │   ├── honeypots.pkl
│   │   └── trajectory.npy
│   └── sample_candidates.json
├── generate_tree.py
├── indexing
│   ├── bm25_builder.py
│   ├── faiss_builder.py
│   ├── feature_store.py
│   ├── honeypot_registry.py
│   └── trajectory_builder.py
├── job_description.md
├── ontology
│   ├── graph_traversal.py
│   ├── query_expander.py
│   └── skill_map.json
├── parsed_job_description.json
├── pipeline
│   ├── candidate_parser.py
│   ├── jd_parser.py
│   ├── runner.py
│   └── schemas.py
├── precompute.py
├── rank.py
├── requirements.txt
├── retrieval
│   ├── __init__.py
│   ├── keyword_path.py
│   ├── ontology_path.py
│   ├── rrf_fusion.py
│   ├── semantic_path.py
│   ├── signal_path.py
│   └── trajectory_path.py
├── scoring
│   ├── behavioral.py
│   ├── career_quality.py
│   ├── composite.py
│   ├── cross_encoder.py
│   ├── honeypot_filter.py
│   ├── skill_match.py
│   └── trajectory.py
├── scripts
│   ├── benchmark_runtime.py
│   ├── inspect_candidates.py
│   └── validate_output.py
├── submission_metadata.yaml
├── tests
│   ├── conftest.py
│   ├── test_e2e.py
│   ├── test_honeypot.py
│   ├── test_reasoning.py
│   ├── test_retrieval.py
│   └── test_scoring.py
├── trust
│   ├── advocate.py
│   ├── reasoning_generator.py
│   ├── skeptic.py
│   └── verdict.py
└── ui
    ├── app.py
    ├── ats_platform.html
    └── components
        ├── candidate_card.py
        └── score_breakdown.py
```
