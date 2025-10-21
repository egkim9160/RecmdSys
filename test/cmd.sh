model_dir="/SPO/Project/RecSys/251020_edit_careergap/models/fold_5/"
index_name="resume_gm_v100_20250917"
board_idx=14731

run_dir="/SPO/Project/HLink/RecmdSys/test/board_${board_idx}"
candidates_dir="candidates"

# 1) candidates 삭제
#rm -rf "${candidates_dir}"

# 2) candidates 생성 (OpenSearch → CSV, 벡터+좌표 포함, LLM 폴백 지오코딩)
python /SPO/Project/HLink/RecmdSys/tools/export_candidates_from_opensearch.py \
  --index_name "${index_name}" \
  --out_dir "${candidates_dir}"

# 3) 공고 파싱 + 04 전처리 + 모델 스코어 산출 (로컬 candidates 사용)
python /SPO/Project/HLink/RecmdSys/tools/hlink_infer_pipeline.py \
  --hlink_idx ${board_idx} \
  --candidates_csv "${candidates_dir}/candidates_ok.csv" \
  --model_json ${model_dir}/xgb_model.json \
  --features_json ${model_dir}/data_info.json \
  --out_dir "${run_dir}"


