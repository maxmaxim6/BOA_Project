import os
import command


folders = ["d_hi_chrome", "d_hi_safari", "l_hi_chrome", "l_hi_ff", "w_hi_chrome", "w_hi_ff", "w_hi_ie"]
files = []

for folder in folders:
    for file in os.listdir(folder):
        if file.endswith(".pcap"):
            files.append(folder + "/" + file)

packets = []
for file in files:
    cmd = "tshark -r " + file + " | wc -l"
    result = command.getstatusoutput(cmd)
    packets.append(result[1])

file = open("packets.txt", "w")
for pack in packets:
    file.write(str(pack) + ",")
file.close()
