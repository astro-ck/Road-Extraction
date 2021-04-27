from subprocess import Popen

cmd1 = "python3 train_ensemble.py 1"
cmd2 = "python3 train_ensemble.py 2"
cmd3 = "python3 train_ensemble.py 3"
cmd4 = "python3 train_ensemble.py 4"
cmd5 = "python3 train_ensemble.py 5"

# cmd1 = "python3 test.py 1"
# cmd2 = "python3 test.py 2"
# cmd3 = "python3 test.py 3"
# cmd4 = "python3 test.py 4"
# cmd5 = "python3 test.py 5"

p1 = Popen(cmd1, shell=True)
p1.wait()
p2 = Popen(cmd2, shell=True)
p2.wait()
p3 = Popen(cmd3, shell=True)
p3.wait()
p4 = Popen(cmd4, shell=True)
p4.wait()
p5 = Popen(cmd5, shell=True)
p5.wait()

