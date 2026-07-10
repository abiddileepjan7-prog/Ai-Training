import subprocess


def run_ollama(command):
    """
    Executes an Ollama command and prints the output.
    """
    print("\n" + "=" * 70)
    print("Running Command:", " ".join(command))
    print("=" * 70)

    try:
        result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
        )

        print("Return Code:", result.returncode)

        if result.stdout:
            print("\nOUTPUT:")
            print(result.stdout)

        if result.stderr:
            print("\nERROR:")
            print(result.stderr)

    except FileNotFoundError:
        print("Error: Ollama is not installed or not added to PATH.")


# ============================================================
# CHANGE THIS MODEL IF REQUIRED
# ============================================================
MODEL = "llama3.2:1b"


# ============================================================
# 1. Ollama Version
# ============================================================
run_ollama(["ollama", "--version"])


# ============================================================
# 2. Ollama Help
# ============================================================
run_ollama(["ollama", "help"])


# ============================================================
# 3. Pull Model
# (Downloads the model if not already available)
# ============================================================
run_ollama(["ollama", "pull", MODEL])


# ============================================================
# 4. List Downloaded Models
# ============================================================
run_ollama(["ollama", "list"])


# ============================================================
# 5. Show Model Information
# ============================================================
run_ollama(["ollama", "show", MODEL])


# ============================================================
# 6. Run the Model
# ============================================================
run_ollama([
    "ollama",
    "run",
    MODEL,
    "What is Artificial Intelligence?"
])


# ============================================================
# 7. Show Running Models
# ============================================================
run_ollama(["ollama", "ps"])


# ============================================================
# 8. Stop the Running Model
# ============================================================
run_ollama(["ollama", "stop", MODEL])