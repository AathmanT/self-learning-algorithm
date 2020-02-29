import subprocess

subprocess.call("python -m cProfile -o output.pstats zoo_gsom.py")
subprocess.call("gprof2dot -f pstats output.pstats | C:\Program Files (x86)\Graphviz2.38\bin\dot.exe -Tpng -o output.png")