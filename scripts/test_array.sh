#!/usr/bin/env bash
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=20:00
#SBATCH --exclude=gpic09,gpic10
#SBATCH --output=./logs/test/test_lora_%A_%a.log
#SBATCH --error=./logs/test/test_lora_%A_%a.log
#SBATCH --job-name=LoRAArray   # LoRAArray / AnySatArray / SoftConArray / LVDArray / DOFAArray / PanopticonArray


set -euo pipefail

CMDS_FILE="${1:-test_commands_dream.txt}"
# CMDS_FILE="${1:-test_commands_mdas.txt}"

# Read the corresponding line (1-based index)
cmd="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CMDS_FILE" || true)"

# If the line is empty or a comment, exit early
if [[ -z "${cmd//[[:space:]]/}" ]] || [[ "$cmd" =~ ^[[:space:]]*# ]]; then
  echo "[$SLURM_ARRAY_TASK_ID] Empty or commented line; nothing to do."
  exit 0
fi

# Extract --name to rename the job (optional but useful for monitoring)
job_name="$(echo "$cmd" | sed -n 's/.*--name[= ]"\?\([^"[:space:]]*\)"?.*/\1/p')"
if [[ -n "${job_name}" ]]; then
  scontrol update JobId="${SLURM_JOB_ID}" JobName="${job_name}" || true
fi

# Create a temporary script to avoid quoting issues
tmp_script="$(mktemp)"
cat > "${tmp_script}" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
EOS
# Append preamble and the actual command
{
  # Print useful job metadata
  echo 'echo "[JOB $SLURM_JOB_ID / TASK $SLURM_ARRAY_TASK_ID] Running..."'
  # Insert the command as-is (preserving double quotes)
  printf "%s\n" "$cmd"
} >> "${tmp_script}"

chmod +x "${tmp_script}"
bash "${tmp_script}"
