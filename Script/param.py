import os

# --------------- Paths ----------------------
bpth = os.getcwd().replace("\\feat", "")
bpth = bpth.replace("\\model", "")
bpth = bpth.replace("\\Script", "")
featpth = bpth + "\\Features"
inpth = bpth + "\\Input"
outpth = bpth + "\\Output"
modellogpth = bpth + "\\Modellog"

# ------------ 7 classes de median relevances ----------------
dclasstorel = {1: 1, 2: 1.33, 3: 1.67, 4: 2, 5: 2.33, 6: 2.67, 7: 3}
dreltoclass = {v: k for k, v in dclasstorel.items()}

# ------------------- split train/test -----------------------
nbtrain = 74067
nball = 240760
