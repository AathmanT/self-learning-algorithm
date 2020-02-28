import subprocess

subprocess.call("python -m cProfile -o outputsdfs.pstats zoo_gsom.py")
subprocess.call("gprof2dot -f pstats outputsdfs.pstats | C:\Program Files (x86)\Graphviz2.38\bin\dot.exe -Tpng -o outputsdfs.png")