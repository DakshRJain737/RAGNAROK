import config
from indexing.honeypot_registry import HoneypotFilter
from indexing.trajectory_builder import TrajectoryAnalyzer
import time
from pipeline.jd_parser import JDParser
from pipeline.candidate_parser import CandidateParser
from pathlib import Path
from indexing.faiss_builder import FaissIndex
from indexing.bm25_builder import BM25Index
from indexing.feature_store import FeatureStore


DATASET_PATH = Path("sample_candidates.json")

time1 = time.perf_counter()


candidate_parser = CandidateParser() 
candidates = candidate_parser.build_candidate_list(DATASET_PATH)
print("Candidates loaded successfully")

honeypot_filter = HoneypotFilter()
honeypot_filter.run_honeypot_filters(candidates)
print("Honeypot run successfully")

trajectory_analyzer = TrajectoryAnalyzer()
trajectory_analyzer.build_all_feature_vector(candidates)
print("Trajectory Analyzer run successfully")

parser = JDParser()
intent = parser.parse(Path("job_description.md"), encode=False)  # encode=False skips model load
print("Job description parser run successfully")

fi = FaissIndex()
fi.build(candidates, save=True)
print("Faiss index built successfully")

bm25 = BM25Index()
bm25.build(candidates, save=True)
print("BM25 index built successfully")

fs = FeatureStore()
matrix = fs.build(candidates, save=True)
print("Feature store run successfully")

time2 = time.perf_counter()
print(time2 - time1)