#!/usr/bin/env bash
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=20:00
#SBATCH --output=./logs/test_alignet_%A_%a.log
#SBATCH --error=./logs/test_alignet_%A_%a.log
#SBATCH --job-name=ALIGNetArray


set -euo pipefail

CMDS_FILE="${1:-test_commands_dream.txt}"
# CMDS_FILE="${1:-test_commands_mdas.txt}"

# Llegeix la línia corresponent (1-based)
cmd="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CMDS_FILE" || true)"

# Si és buida o comentari, surt
if [[ -z "${cmd//[[:space:]]/}" ]] || [[ "$cmd" =~ ^[[:space:]]*# ]]; then
  echo "[$SLURM_ARRAY_TASK_ID] Línia buida o comentada; res a fer."
  exit 0
fi

# Extreu --name per renombrar la tasca (opcional però útil)
job_name="$(echo "$cmd" | sed -n 's/.*--name[= ]"\?\([^"[:space:]]*\)"?.*/\1/p')"
if [[ -n "${job_name}" ]]; then
  scontrol update JobId="${SLURM_JOB_ID}" JobName="${job_name}" || true
fi

# Crea un script temporal per evitar problemes de quoting
tmp_script="$(mktemp)"
cat > "${tmp_script}" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
EOS
# Afegeix el preàmbul i la comanda real
{
  # Imprimeix metadata útil
  echo 'echo "[JOB $SLURM_JOB_ID / TASK $SLURM_ARRAY_TASK_ID] Executant..."'
  # Insereix la comanda tal qual (incloses cometes dobles)
  printf "%s\n" "$cmd"
} >> "${tmp_script}"

chmod +x "${tmp_script}"
bash "${tmp_script}"