# PlatformIO pre-script for env:sim: adds SDL2 include/lib paths (works on macOS/Homebrew and Linux).
Import("env")
import subprocess
prefix = subprocess.check_output(["sdl2-config", "--prefix"]).decode().strip()
env.Append(
    CPPPATH=[prefix + "/include", prefix + "/include/SDL2"],
    LIBPATH=[prefix + "/lib"],
    LIBS=["SDL2"],
)
