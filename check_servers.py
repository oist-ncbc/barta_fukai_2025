import subprocess
import multiprocessing

# Define servers
SERVERS = [f"tomas-barta@ncbc{i:02d}.oist.jp" for i in range(1, 11)]

def format_memory(mem_mb):
    """Convert memory in MB to a human-readable format."""
    mem_mb = int(mem_mb)
    if mem_mb >= 1024:
        return f"{mem_mb / 1024:.1f} GB"
    return f"{mem_mb} MB"

def get_server_usage(server):
    """Fetch CPU and memory usage from a remote Linux server."""
    try:
        command = """
        top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}';
        free -m | awk 'NR==2{printf "%s %s", $2, $3}';
        """
        
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", server, command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode == 0:
            output = result.stdout.strip().split("\n")
            if len(output) == 2:
                cpu_usage = output[0]  # CPU Load
                total_mem, used_mem = output[1].split()  # Memory
                total_mem = format_memory(total_mem)
                used_mem = format_memory(used_mem)
                return (server, cpu_usage, total_mem, used_mem)
            else:
                return (server, "Error", "Error", "Error")
        else:
            return (server, "Error", "Error", "Error")

    except Exception as e:
        return (server, "Error", "Error", "Error")


def main():
    print("\n🔎 Checking SSH servers...\n")
    print("\n📊 Server Usage Summary:\n")
    print(f"{'Server':<35}{'CPU Load':<12}{'Total Mem':<12}{'Used Mem':<12}")
    print("=" * 75)

    # Run SSH queries in parallel
    with multiprocessing.Pool(processes=5) as pool:
        results = pool.map(get_server_usage, SERVERS)

    # Display results
    for server, cpu, total_mem, used_mem in results:
        print(f"{server:<35}{cpu:<12}{total_mem:<12}{used_mem:<12}")

if __name__ == "__main__":
    main()
