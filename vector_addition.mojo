from max.gpu.host import DeviceContext
from std.sys import has_accelerator
from std.gpu import block_idx, thread_idx

def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
    else:
        var ctx = DeviceContext()
        print("Found GPU:", ctx.name())
        print("block_idx\t\tthread_idx")
        print("x\ty\tz", "x\ty\tz", sep="\t")
        print("-" * 20, "-" * 20, sep="\t")
        ctx.enqueue_function[print_threads](
            grid_dim=(2, 2, 1),
            block_dim=(16, 4, 2)
        )
        ctx.synchronize()
        print("Program finished")

def print_threads():
    """Print thread IDs."""

    print(
        block_idx.x,
        block_idx.y,
        block_idx.z,
        thread_idx.x,
        thread_idx.y,
        thread_idx.z,
        sep="\t",
    )
