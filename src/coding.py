import subprocess
import uuid
import re


def extract_code(response: str) -> str:
    """Extract code from markdown fences if present. 

    Like this:
    ```python
    print("Hello, World!")
    ```
    
    """
    match = re.search(r"```(?:python)?\s*\n(.+?)```", response, re.DOTALL)
    return match.group(1).strip() if match else response.strip()


def run_python_sandboxed(code: str, timeout: int = 30, max_output_bytes: int = 1_048_576) -> dict:
    """
    Run untrusted Python code safely in an isolated Docker container.

    Think of Docker as a lightweight virtual machine. This function spins up
    a tiny, disposable Linux environment, runs the code inside it, and tears
    it down afterwards. The real machine (your laptop/server) is never at risk.

    Safety measures applied:
        - No network access         → code can't phone home or download anything
        - 512 MB RAM cap            → code can't eat all your memory
        - 1 CPU core max            → code can't hog your processor
        - 50 process limit          → blocks fork-bombs (spawning infinite processes)
        - Read-only filesystem      → code can't write outside of /tmp
        - Unprivileged user         → code runs as a nobody, not as root
        - No privilege escalation   → code can't promote itself to root
        - Timeout                   → long-running / infinite loops get killed
        - Output cap                → prevents memory exhaustion from huge prints

    Args:
        code:             The Python source code to execute.
        timeout:          Max wall-clock seconds before the container is killed.
        max_output_bytes: Max bytes captured from stdout/stderr (default 1 MB).

    Returns:
        dict with keys:
            status    – "success", "failed", "timeout", or "system_error"
            stdout    – captured standard output  (truncated if over the cap)
            stderr    – captured standard error   (truncated if over the cap)
            exit_code – 0 on success, non-zero on failure, -1 on timeout/error
    """

    # Give every container a unique name so we can always find and kill it,
    # even if the Python process that started it crashes.
    container_name = f"sandbox_{uuid.uuid4().hex}"

    cmd = [
        "docker", "run",
        "--rm",                                    # auto-remove container when it exits
        f"--name={container_name}",                # unique name for cleanup
        "--network=none",                          # no internet access
        "--memory=512m",                           # RAM ceiling
        "--cpus=1",                                # CPU ceiling
        "--pids-limit=50",                         # max number of processes
        "--read-only",                             # filesystem is immutable
        "--user=1000:1000",                        # run as unprivileged user
        "--security-opt=no-new-privileges:true",   # can't escalate to root
        "--tmpfs=/tmp:size=16m,mode=1777",         # small writable scratch space
        "--shm-size=16m",                          # cap shared memory
        "--ulimit", "nofile=64:64",                # cap open file descriptors
        "-i",                                      # keep stdin open so we can pipe code in
        "python:3.12-slim",                        # the Python image to use
        "python3", "-"                             # tell Python to read code from stdin
    ]

    try:
        # We pipe the code through stdin instead of passing it as a command-line
        # argument. This avoids problems with quotes, special characters, and
        # argument length limits — important because LLM-generated code can
        # contain anything.
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Feed the code in and wait for the container to finish.
        # communicate() closes stdin after writing, which signals Python
        # inside the container to start executing.
        raw_stdout, raw_stderr = proc.communicate(
            input=code.encode(),
            timeout=timeout,
        )

        # Enforce the output cap — a script that prints gigabytes would
        # otherwise eat all our memory before the timeout fires.
        stdout = raw_stdout[:max_output_bytes].decode(errors="replace").strip()
        stderr = raw_stderr[:max_output_bytes].decode(errors="replace").strip()

        return {
            "status": "success" if proc.returncode == 0 else "failed",
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": proc.returncode,
        }

    except subprocess.TimeoutExpired:
        # The code ran too long.  Kill the container immediately so it
        # stops consuming CPU/RAM on the host.
        proc.kill()                # kill the local subprocess
        proc.wait()               # reap the zombie process
        subprocess.run(           # force-remove the Docker container
            ["docker", "rm", "-f", container_name],
            capture_output=True,
        )
        return {
            "status": "timeout",
            "stdout": "",
            "stderr": f"Execution timed out after {timeout} seconds.",
            "exit_code": -1,
        }

    except Exception as e:
        # Something unexpected went wrong (Docker not installed, permissions
        # issue, etc.).  Clean up the container just in case it was created.
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
        )
        return {
            "status": "system_error",
            "stdout": "",
            "stderr": f"System failure: {str(e)}",
            "exit_code": -1,
        }