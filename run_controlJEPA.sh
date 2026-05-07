LBD_LIST=(0.01 0.02 0.05 0.1)
G_LIST=(0.8 0.9 0.95 0.98)
T_LIST=(1e-3 1e-4 1e-5 0)

for lbd in "${LBD_LIST[@]}"; do
  for g in "${G_LIST[@]}"; do
    for t in "${T_LIST[@]}"; do
      LBD_CONTROL="$lbd" TUBE_GAMMA="$g" TUBE_TAU="$t" \
        bash run_stp.sh
    done
  done
done